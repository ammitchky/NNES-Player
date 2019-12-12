import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import time

class FceuxNesEmulatorEnvironment:
    def __init__(self, rom='none', save_state='none', initial_step=0):
        # Which ROM is currently being emulated?
        self.game_name = '***'
        self.game_id = '-1'
        # How many steps have been made? Should the environment end after a number of steps?
        self.time_step = 0
        self.max_step = 1000
        # What is the reward and penalty for each step, failure, and success?
        self.transaction_penalty = -1.0
        self.fail_penalty = -250.0
        self.goal_reward = 0.0
        self.score_multiplier = 1.0
        self.timeout_penalty = 0.0
        # Specifies which ROM and/or save state should be loaded next frame.
        self.load_rom = rom
        self.load_state = save_state
        # Repeatedly used action settings
        self.no_action = ['false', 'false', 'false', 'false', 'false', 'false', 'false', 'false']
        # The core features of the current step, used by the NN
        self.action = self.no_action
        self.state = None
        self.reward = 0
        # Used as a timestamp to validate text files
        self.validation_step = initial_step
        # Amount of time to delay between each check of a txt File
        self.sleep_amount = 0.1
        # Up-to-date In-Game Score (Points)
        self.score = 0
        # Load specified state & action
        self.write_action()


    def reset(self, save_state='1'):
        # Reset State and return the Initial State
        self.action = self.no_action
        self.load_rom = 'none'
        self.load_state = save_state
        return self.write_action()

    def step(self, action):
        # Perform Action and Get State
        self.action = action
        state = self.write_action()
        # Get Reward / Done Status / Info
        reward, done = self.read_rewards()
        info = ""
        return state, reward, done, info

    def read_state(self):
        updated = False
        # Wait until File is Updated
        while not updated:
            # Open the File
            file = open("screen.txt", "r")
            # Make List out of Lines of File
            lines = file.readlines()
            print(self.validation_step)
            if lines is not False and int(lines[0]) >= self.validation_step:
                updated = True
            else:
                print("Not Updated")
                file.close()
                time.sleep(self.sleep_amount)
        # Loop through each Line, and format
        line_index = 0
        screen = []
        scr_r = []
        scr_g = []
        scr_b = []
        scr_x = []
        for line in lines:
            if line_index != 0:
                value = line.split()
                # broke up input into 4 different channels
                scr_r.append(int(value[0]))
                scr_g.append(int(value[1]))
                scr_b.append(int(value[2]))
                scr_x.append(int(value[3]))
            line_index += 1
        screen.append(scr_r)
        screen.append(scr_g)
        screen.append(scr_b)
        screen.append(scr_x)
        # Close the File
        file.close()
        # Update and Return current State
        # Convert pixel data into tensor that will be accepted by Conv2d layer
        self.state = torch.FloatTensor(screen).reshape(4, 28, 32).unsqueeze(0)
        return self.state

    def write_action(self):
        # Open the File
        file = open("input.txt", "w")
        # Prepare Lines of File
        contents = self.action
        contents.append(self.load_state)
        contents.append(self.load_rom)
        # Create File Contents String
        content_string = str(self.validation_step) + '\n'
        for line in contents:
            content_string = content_string + line + '\n'
        # Write to the file
        file.write(content_string)
        # Close the File
        file.close()
        # Reset Action Variables
        self.load_rom = 'none'
        self.load_state = 'none'
        self.action = [] # None
        # Increment Validation Step
        self.validation_step += 1
        # Return the New State (After the Action is Performed)
        return self.read_state()

    def read_rewards(self):
        updated = False
        # Wait until File is Updated
        while not updated:
            # Open the File
            file = open("variables.txt", "r")
            # Make List out of Lines of File
            lines = file.readlines()
            if lines is not False and int(lines[0]) >= self.validation_step:
                updated = True
            else:
                file.close()
                time.sleep(self.sleep_amount)
        # Set Default Return Values
        reward = 0
        done = False
        # Save last Frame's Score
        previous_score = self.score
        # Loop through each Line, and check contents
        line_index = 0
        for line in lines:
            if line_index != 0:
                elements = line.split()
                if elements[0] == 'SCORE':
                    self.score = int(elements[1])
                elif elements[0] == 'GAME_ID':
                    self.game_id = elements[1]
                elif elements[0] == 'GAME_NAME':
                    self.game_name = elements[1]
                elif elements[0] == 'GAME_OVER':
                    if elements[1] == 'true':
                        done = True
                        reward += self.fail_penalty
                elif elements[0] == 'VICTORY':
                    if elements[1] == 'true':
                        done = True
                        reward += self.goal_penalty
            line_index += 1
        # Close the File
        file.close()
        # Apply reward based on Score Increase
        reward += (self.score - previous_score) * self.score_multiplier
        # Apply Transactional Penalty
        reward += self.transaction_penalty
        # Check for Timeout
        if not done and self.time_step >= self.max_step:
            done = True
            reward += self.timeout_penalty
        # Set the Reward and Return
        self.reward = reward
        return reward, done
    
    
class CliffWalkingEnv():
    # Cliff walk environment with OpenAI Gym like interface
    def __init__(self, width=10, height=6):
        self.width = width
        self.height = height
        self.x = 0
        self.y = height-1
        self.move_count = 0
        self.max_step = 50
        self.transaction_penalty = -5.0
        self.fall_penalty = -250
        self.edge_penalty = -250
        self.goal_reward = 0.0
        self.no_fall = False

        self.map = [["." for j in range(width)] for i in range(height)]

    def reset(self):
        self.x = 0
        self.y = self.height-1
        self.move_count = 0
        return self.y * self.width + self.x

    def step(self, action):
        self.move_count += 1
        if action == 0:
            # left
            self.x = self.x-1
        elif action == 1:
            # up
            self.y = self.y-1
        elif action == 2:
            # right
            self.x = self.x+1
        elif action == 3:
            # down
            self.y = self.y+1

        reward = self.transaction_penalty
        done = False

        if not self.no_fall:
            if self.x < 0 or self.x >= self.width or self.y < 0 or self.y >= self.height:
                done = True
                reward = self.edge_penalty * (self.width-self.x+1)/self.width

        self.x = max(0, self.x)
        self.y = max(0, self.y)
        self.x = min(self.width - 1, self.x)
        self.y = min(self.height - 1, self.y)

        if self.y == self.height-1 and self.x > 0:
            reward = self.fall_penalty * (self.width-self.x+1)/self.width

            # bounce the agent back to a valid state, this is critical for actor_critic,
            # since it can't update the value of the unused state but will ask for its value via next_state.
            if action == 3:
                self.y = self.y-1
            elif action == 2:
                self.x = self.x-1

            done = True
            if self.x == self.width-1:
                reward = self.goal_reward

        if self.move_count > 50:
            done = True

        state = self.y * self.width + self.x
        return state, reward, done, {}


class PolicyEstimator(torch.nn.Module):
    def __init__(self, device=torch.device("cpu"), dtype=torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.state_size = 6*10
        self.action_size = 8
        self.h1_size = 32

        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=4, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.linear1 = torch.nn.Linear(64*8*10, 512)
        self.linear2 = torch.nn.Linear(512, self.action_size)

    def forward(self, state):
        state = torch.relu(self.conv1(state))
        state = torch.relu(self.conv2(state))
        state = torch.relu(self.conv3(state))

        state = state.view(-1, 64*8*10)

        state = torch.relu(self.linear1(state))
        action = torch.sigmoid(self.linear2(state))
        return action


class ValueEstimator(torch.nn.Module):
    def __init__(self, device=torch.device("cpu"), dtype=torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.state_size = 6*10
        self.h1_size = 32

        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=4, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.linear1 = torch.nn.Linear(64 * 8 * 10, 512)
        self.linear2 = torch.nn.Linear(512, 1)

    def forward(self, state):
        state = torch.relu(self.conv1(state))
        state = torch.relu(self.conv2(state))
        state = torch.relu(self.conv3(state))

        state = state.view(-1, 64 * 8 * 10)

        #state = torch.relu(self.linear1(state))
        h1 = self.linear1(state).clamp(min=0)
        value = self.linear2(h1)
        return value

def toAction(actionProp):
    nextAction = []
    for est in actionProp:
        if est > 0.5:
            nextAction.append('true')
        else:
            nextAction.append('false')
    return nextAction

def reinforce(device="cpu"):
    discount_factor = 0.7
    lr = 1e-2
    save_path = "reinforce/"
    train = True
    env = CliffWalkingEnv()

    policy_estimator = PolicyEstimator()
    value_estimator = ValueEstimator()

    policy_optimizer = torch.optim.Adam(policy_estimator.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_estimator.parameters(), lr=lr)

    stats = {}
    stats["episode_rewards"] = []
    stats["episode_length"] = []

    if os.path.isfile(save_path+"policy_estimator"):
        policy_estimator.load_state_dict(torch.load(save_path+"policy_estimator"))
        print("Policy estimator loaded")
    if os.path.isfile(save_path+"value_estimator"):
        value_estimator.load_state_dict(torch.load(save_path+"value_estimator"))
        print("Value estimator loaded")

    #for name in policy_estimator.state_dict():
    #    print(name, " ", policy_estimator.state_dict()[name])

    if train:
        avg_reward = 0
        avg_length = 0
        for i in range(200000):
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0
            rewards = []
            states = []
            actions = []
            for t in itertools.count():
                action_probs = policy_estimator(torch.tensor(state, dtype=torch.long))
                action = torch.multinomial(action_probs, 1)
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                episode_reward += reward
                episode_length = t
                if done:
                    break
                state = next_state

            stats["episode_rewards"].append(episode_reward)
            stats["episode_length"].append(episode_length)

            #print("Episode reward: {}".format(episode_reward))
            #print("Episode length: {}".format(episode_length))

            for t in range(len(rewards)):
                total_reward = np.sum([discount_factor**j * reward for j, reward in enumerate(rewards[t:])])
                state = torch.tensor(states[t], dtype=torch.long)
                action = torch.tensor(actions[t], dtype=torch.long)
                baseline_value = value_estimator(state)

                advantage = total_reward - baseline_value

                # update value estimator
                value_loss = (total_reward-baseline_value) ** 2
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                # update policy estimator
                action_onehot = policy_estimator(state)
                action_prob = action_onehot[action]
                policy_loss = -torch.log(action_prob) * advantage.detach()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

            avg_reward += episode_reward
            avg_length += episode_length
            if i % 200 == 0:
                print("Average episode reward: {}".format(avg_reward / 200))
                print("Average episode length: {}".format(avg_length / 200))
                avg_reward = 0
                avg_length = 0

            if i % 10000 == 0:
                print("Saving model...")
                torch.save(policy_estimator.state_dict(), save_path + "policy_estimator")
                torch.save(value_estimator.state_dict(), save_path + "value_estimator")
                draw_policy("reinforce_{}".format(i), policy_estimator, value_estimator, device=device)

    draw_policy("reinforce_final", policy_estimator, value_estimator, device=device)


def actor_critic(device="cpu"):
    discount_factor = 0.7
    lr = 1e-3
    random_chance = 0.04
    save_path = "actor_critic/"
    train = True

    fenv = FceuxNesEmulatorEnvironment()
    policy_estimator = PolicyEstimator()
    value_estimator = ValueEstimator()

    policy_optimizer = torch.optim.Adam(policy_estimator.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_estimator.parameters(), lr=lr)

    if os.path.isfile(save_path+"policy_estimator"):
        policy_estimator.load_state_dict(torch.load(save_path+"policy_estimator"))
        print("Policy estimator loaded")
    if os.path.isfile(save_path+"value_estimator"):
        value_estimator.load_state_dict(torch.load(save_path+"value_estimator"))
        print("Value estimator loaded")

    if train:
        avg_reward = 0
        avg_length = 0
        for i in range(200000):
            state = fenv.reset()
            episode_reward = 0.0
            episode_length = 0
            rewards = []
            states = []
            actions = []
            for t in itertools.count():
                action_probs = policy_estimator(state)
                print(action_probs)
                if np.random.uniform() < random_chance:
                    action = torch.FloatTensor(8).uniform_(0, 1).detach()
                else:
                    action = action_probs[0]
                print(action)
                true_act = toAction(action)
                next_state, reward, done, _ = fenv.step(true_act)
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                episode_reward += reward
                episode_length = t

                next_value = value_estimator(next_state)
                target_value = reward + discount_factor * next_value
                predict_value = value_estimator(state)

                advance = target_value.detach() - predict_value

                value_loss = (target_value.detach() - predict_value) ** 2
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                action_prob = 1.0
                for act in action_probs[0]:
                    if act > 0.5:
                        action_prob = action_prob * act
                    else:
                        action_prob = action_prob * (1 - act)
                #action_prob = action_probs[action]
                policy_loss = -torch.log(action_prob) * advance.detach()
                # print(policy_loss)
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                if done:
                    break
                state = next_state

            #print("Episode reward: {}".format(episode_reward))
            #print("Episode length: {}".format(episode_length))

            avg_reward += episode_reward
            avg_length += episode_length
            if i % 200 == 0:
                print("Average episode reward: {}".format(avg_reward/200))
                print("Average episode length: {}".format(avg_length/200))
                avg_reward = 0
                avg_length = 0

            if i % 2000 == 0:
                print("Saving model...")
                torch.save(policy_estimator.state_dict(), save_path + "policy_estimator")
                torch.save(value_estimator.state_dict(), save_path + "value_estimator")
                draw_policy("actor_critic_{}".format(i), policy_estimator, value_estimator, device=device)

    draw_policy("actor_critic_final", policy_estimator, value_estimator, device=device)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def draw_policy(name, policy_model, value_model, device="cpu"):
    width = 10
    height = 6
    cells = [[plt.text(x, y, "({},{})".format(x, y)) for x in range(width)] for y in range(height)]
    action_probs = []
    pred_values = []

    plt.clf()
    plt.figure(figsize=(20, 12))
    plt.xlim(-0.6, width-0.4)
    plt.ylim(-0.6, height-0.4)

    for y in range(height):
        action_probs.append([])
        pred_values.append([])
        for x in range(width):
            state = torch.tensor(y * width + x, device=device)
            action_probs[y].append(policy_model(state).detach().cpu().numpy().tolist())
            pred_values[y].append(value_model(state).detach().cpu().numpy()[0])
            normalized_action_prob = normalized(action_probs[y][x]).reshape(-1)

            plt.text(x+0.1, height-y-1+0.1, "({:.2f})".format(pred_values[y][x]))
            facecolor = "b"
            if y == height-1 and x > 0:
                facecolor = "r"
                if x == width-1:
                    facecolor = "g"
            plt.arrow(x, height-y-1, -normalized_action_prob[0]/2, 0, width=0.03, facecolor=facecolor)
            plt.arrow(x, height-y-1, 0, normalized_action_prob[1]/2, width=0.03, facecolor=facecolor)
            plt.arrow(x, height-y-1, normalized_action_prob[2]/2, 0, width=0.03, facecolor=facecolor)
            plt.arrow(x, height-y-1, 0, -normalized_action_prob[3]/2, width=0.03, facecolor=facecolor)

    print(action_probs)
    print(pred_values)

    plt.savefig(name+".png")


if __name__ == "__main__":
    file = open("screen.txt", "w")
    file.write('0')
    # Close the File
    file.close()
    file = open("variables.txt", "w")
    file.write('0')
    # Close the File
    file.close()
    #reinforce()
    #fenv = FceuxNesEmulatorEnvironment()
    #pnet = PolicyEstimator()
    #print(fenv.read_state())
    #print(torch.FloatTensor(8).uniform_(0, 1).detach())
    #print(toAction(pnet(fenv.read_state())))
    actor_critic()
