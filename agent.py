import torch
import torch.distributions.categorical as cat
import numpy as np
import itertools
import os
import time
import subprocess


class FceuxNesEmulatorEnvironment:
    def __init__(self, rom='none', save_state='none', initial_step=0):
        # Which ROM is currently being emulated?
        self.game_name = '***'
        self.game_id = '-1'
        # How many steps have been made? Should the environment end after a number of steps?
        self.time_step = 0
        self.max_step = 1000
        # What is the reward and penalty for each step, failure, and success?
        self.transaction_penalty = 0.0
        self.fail_penalty = -250.0
        self.goal_reward = 0.0
        self.score_multiplier = 2.0
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
        self.sleep_amount = 0.05
        # Up-to-date In-Game Score (Points)
        self.score = 0
        # Load specified state & action
        #self.write_action()


    def reset(self, save_state='1'):
        # Reset State and return the Initial State
        self.time_step = 0
        self.score = 0
        self.action = self.no_action
        self.load_rom = 'none'
        self.load_state = save_state
        temp = self.write_action()
        if os.path.isfile("variables.txt"):
            os.remove("variables.txt")
        return temp

    def step(self, action):
        # Perform Action and Get State
        self.action = action
        state = self.write_action()
        self.time_step = self.time_step + 1
        # Get Reward / Done Status / Info
        reward, done = self.read_rewards()
        info = ""
        return state, reward, done, info

    def read_state(self):
        updated = False
        # Wait until File is Updated
        while not updated:
            # Open the File
            #time.sleep(0.04)
            if os.path.isfile("ram.txt"):
                try:
                    file = open("ram.txt", "r")
                except PermissionError:
                    print("minor issue, try again")
                    continue
                # Make List out of Lines of File
                lines = file.readlines()
                print(self.validation_step)
                if lines and int(lines[0]) >= self.validation_step:
                    updated = True
                else:
                    # print("Not Updated")
                    file.close()
                    time.sleep(self.sleep_amount*3)
            else:
                # print("Not Updated")
                time.sleep(self.sleep_amount)
        # Loop through each Line, and format
        line_index = 0
        ram = []
        for line in lines:
            if line_index != 0:
                ram.append(int(line.split()[0]))
            line_index += 1
        # Close the File
        file.close()
        if os.path.isfile("ram.txt"):
            os.remove("ram.txt")
        # Update and Return current State
        # Convert pixel data into tensor that will be accepted by Conv2d layer
        self.state = torch.FloatTensor(ram)
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
            #time.sleep(0.04)
            if os.path.isfile("variables.txt"):
                try:
                    file = open("variables.txt", "r")
                except PermissionError:
                    print("minor issue, try again")
                    continue
                # Make List out of Lines of File
                lines = file.readlines()
                if lines and int(lines[0]) >= self.validation_step:
                    updated = True
                else:
                    file.close()
                    time.sleep(self.sleep_amount)
                    print("test2")
            else:
                print("test")
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
        if os.path.isfile("variables.txt"):
            os.remove("variables.txt")
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


class PolicyEstimator(torch.nn.Module):
    def __init__(self, device=torch.device("cpu"), dtype=torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.state_size = 6*10
        self.action_size = 256
        self.h1_size = 32

        self.linear1 = torch.nn.Linear(2048, 1024)
        self.linear2 = torch.nn.Linear(1024, 512)
        self.linear3 = torch.nn.Linear(512, 256)
        self.linear4 = torch.nn.Linear(256, self.action_size)

    def forward(self, state):
        state = torch.relu(self.linear1(state))
        state = torch.relu(self.linear2(state))
        state = torch.relu(self.linear3(state))

        action = (self.linear4(state).clamp(min=0.01))
        return action


class ValueEstimator(torch.nn.Module):
    def __init__(self, device=torch.device("cpu"), dtype=torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.state_size = 6*10
        self.h1_size = 32

        self.linear1 = torch.nn.Linear(2048, 1024)
        self.linear2 = torch.nn.Linear(1024, 512)
        self.linear3 = torch.nn.Linear(512, 256)
        self.linear4 = torch.nn.Linear(256, 1)

    def forward(self, state):
        state = torch.relu(self.linear1(state))
        state = torch.relu(self.linear2(state))

        #state = torch.relu(self.linear1(state))
        h1 = self.linear3(state).clamp(min=0)
        value = self.linear4(h1)
        return value


def toAction(actionProp):
    nextAction = '{0:08b}'.format(int(actionProp.item()))
    actToString = []
    for est in nextAction:
        if int(est) == 1:
            actToString.append('true')
        else:
            actToString.append('false')
    return actToString


def actor_critic(device="cpu"):
    discount_factor = 0.7
    lr = 1e-3
    random_chance = 0.05
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
                #print(action_probs)
                if np.random.uniform() < random_chance:
                    action = torch.FloatTensor(1).random_(0, 255).detach()[0]
                else:
                    action = cat.Categorical(action_probs).sample().detach()
                #print(action)
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

                m = cat.Categorical(action_probs)
                #action_prob = action_probs[action]
                policy_loss = -m.log_prob(action) * advance.detach()
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

            # print("Average reward: {}".format(avg_reward/(i+1)))
            avg_reward = 0
            avg_length = 0

            print("Saving model...")
            torch.save(policy_estimator.state_dict(), save_path + "policy_estimator")
            torch.save(value_estimator.state_dict(), save_path + "value_estimator")
            #draw_policy("actor_critic_{}".format(i), policy_estimator, value_estimator, device=device)

    #draw_policy("actor_critic_final", policy_estimator, value_estimator, device=device)


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    # rom_path = '/rom/Pac-Man (U) [!].nes'
    subprocess.Popen([path + "/fceux-2.2.3-win32/fceux.exe"])
    # subprocess.Popen(['fceux C:/Users/Owner/Documents/GitHub/NNES-Player/roms/Pac-Man (U) [!].nes'])
    # subprocess.Popen('fceux -lua bridge.lua', info=False)
    actor_critic(device="cuda")
