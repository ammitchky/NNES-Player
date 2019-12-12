-- Read Inputs from File
function readInputFile(fileName, stepNum)
	local file = io.open(fileName, 'r')
	local inputTable = {}
	local commandTable = {}
	if file ~= nil then
		io.input(file)
		local lineNum = 1
		local currentStep = 0
		local value = false
		for line in file:lines() do
			if lineNum == 1 then
				currentStep = tonumber(line)
				if currentStep < stepNum then
					return nil
				else
					print('Updated File! Fetching Input (Step #' .. stepNum .. ')')
				end
			elseif lineNum < 10 then
				if line == 'false' then 
					value = false 
				else 
					value = true 
				end
				if lineNum == 2 then 
					inputTable['A'] = value
				elseif lineNum == 3 then 
					inputTable['up'] = value
				elseif lineNum == 4 then 
					inputTable['left'] = value
				elseif lineNum == 5 then 
					inputTable['B'] = value
				elseif lineNum == 6 then 
					inputTable['select'] = value
				elseif lineNum == 7 then 
					inputTable['right'] = value
				elseif lineNum == 8 then 
					inputTable['down'] = value
				elseif lineNum == 9 then 
					inputTable['start'] = value end
			else
				if lineNum == 10 then 
					commandTable['state'] = line 
				elseif lineNum == 11 then 
					commandTable['rom'] = line 
				end
			end
			lineNum = lineNum + 1
		end
		io.close(file)
	else
		return nil
	end
	return {commandTable, inputTable}
end

-- Write to File
function writeFile(fileName, writeTable, numFields, stepNum)
	local file = io.open(fileName, 'w')
	if file ~= nil then
		io.output(file)
		io.write(stepNum .. '\n')
		for key,value in pairs(writeTable) do
			io.write(value .. ' ')
			if (key%numFields) == 0 then
				io.write('\n')
			end
		end
		io.close()
	end
end

-- Return a table of all pixel values of the current screen
-- 1=Red, 2=Blue, 3=Green, 4=Palette
function getScreenTable()
	local rTable = {}
	local screenMinY = 8
	local screenMinX = 0
	local screenMaxY = 231
	local screenMaxX = 255
	for y=screenMinY, screenMaxY, 8 do
		for x=screenMinX, screenMaxX, 8 do
			local r,g,b,palette = emu.getscreenpixel(x, y, false)
			table.insert(rTable, r)
			table.insert(rTable, g)
			table.insert(rTable, b)
			table.insert(rTable, palette)
		end
	end
	return rTable
end

-- Find entirety of games RAM
function RAMdump()
	local rTable = {}
	local ramMin = 0x0000
	local ramMax = 0x07FF
	for address=ramMin, ramMax, 1 do
		table.insert(rTable, memory.readbyte(address))
	end
	return rTable
end

-- Convert ASCII number to Decimal number
function processNumberASCII(ascii)
	if ascii > 48 and ascii < 58 then
		return ascii - 48
	else
		return 0
	end	
end

-- Find a Weighted Sum of Memory Values
function memorySum(bias, mult, address, iterations, delta)
	local total = bias
	iterations = iterations - 1
	for i=0,iterations do
		total = total + (processNumberASCII(memory.readbyte(address)) * mult)
		address = address + 1
		mult = mult * delta
	end
	return total
end

-- Concatenate a series of Memory Values
function memoryConcat(init, address, iterations)
	local str = init
	iterations = iterations - 1
	for i=0,iterations do
		str = str .. string.format("%x", memory.readbyte(address))
		address = address + 1
	end
	return str
end

-- Create a table representing Game Variables
function getMemoryValues()
	local rTable = {}
	
	-- GAME ID
	local game_id = memoryConcat("", 0xFFE0, 16)
	table.insert(rTable, 'GAME_ID')
	table.insert(rTable, game_id)
	
	-- GAME NAME
	local game_name = "Unknown"
	if game_id == "ffffffffffffffffffffffffffffffff" then
		game_name = "Pac-Man"
	elseif game_id == "ffffffffffffffffff4d4554524f4944" then
		game_name = "Metroid"
	elseif game_id == "1e1f1f1e1d1c1a181614151616171718" then
		game_name = "Super Mario Bros."
	elseif game_id == "025050500250025656560256028554C4" then
		game_name = "Balloon Fight"
	end
	table.insert(rTable, 'GAME_NAME')
	table.insert(rTable, game_name)
	
	-- SCORE
	local score = -1
	if game_name == "Pac-Man" then
		score = memorySum(0, 100000, 0x0240, 5, .1)
	end
	table.insert(rTable, 'SCORE')
	table.insert(rTable, score)
	
	-- GAME OVER
	table.insert(rTable, 'GAME_OVER')
	table.insert(rTable, 'false')
	
	-- VICTORY
	table.insert(rTable, 'VICTORY')
	table.insert(rTable, 'false')
	
	-- LIVES
	local lives = 0
	
	-- PLAYER X
	local px = 0
	
	-- PLAYER Y
	local pxy = 0
	return rTable
end

-- Initialize Variables
-- How many frames have been advanced
timeStep = 0
-- Which Controller to Write to
controllerPort = 1

-- Main Loop
while true do
	-- Read input data from file
	inputTable = readInputFile('input.txt', timeStep)
	
	-- Check to see if the Input File has been updated
	if inputTable == nil then
		-- Input File not updated
	else
		-- Apply Joypad Input, Advance Time Step,
		print(inputTable[2])
		joypad.set(controllerPort, inputTable[2])
		timeStep = timeStep + 1
		-- Load State and/or ROM if specified
		if (inputTable[1])['rom'] ~= "none" and (inputTable[1])['rom'] ~= nil then
			--os.execute()
			print("Load ROM: " .. (inputTable[1])['rom'])
		end
		if (inputTable[1])['state'] ~= "none" and (inputTable[1])['state'] ~= nil then
			savestate.load(savestate.create(tonumber((inputTable[1])['state'])))
			print("Load State: " .. (inputTable[1])['state'])
		end
		
		-- Advance the Frame
		emu.frameadvance()
		
		-- Read the screen's pixels and write to file
		print('Write to Files')
		writeFile('ram.txt', RAMdump(), 1, timeStep)
		writeFile('screen.txt', getScreenTable(), 4, timeStep)
		writeFile('variables.txt', getMemoryValues(), 2, timeStep)
	end
end