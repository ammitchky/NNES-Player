-- Read Inputs from File
function readFile(fileName)
	local file = io.open(fileName, 'r')
	local inputTable = {}
	if file ~= nil then
		io.input(file)
		local lineNum = 1
		local value = false
		for line in file:lines() do
			if line == 'false' then value = false else value = true end
			if lineNum == 1 then inputTable['A'] = value
			elseif lineNum == 2 then inputTable['up'] = value
			elseif lineNum == 3 then inputTable['left'] = value
			elseif lineNum == 4 then inputTable['B'] = value
			elseif lineNum == 5 then inputTable['select'] = value
			elseif lineNum == 6 then inputTable['right'] = value
			elseif lineNum == 7 then inputTable['down'] = value
			elseif lineNum == 8 then inputTable['start'] = value end
			lineNum = lineNum + 1
		end
	end
	return inputTable
end

-- Write to File
function writeFile(fileName, writeTable, numFields)
	local file = io.open(fileName, 'w')
	if file ~= nil then
		io.output(file)
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
	if game_id = "ffffffffffffffffffffffffffffffff" then
		game_name = "Pac-Man"
	elseif game_id = "ffffffffffffffffff4d4554524f4944" then
		game_name = "Metroid"
	elseif game_id = "1e1f1f1e1d1c1a181614151616171718" then
		game_name = "Super Mario Bros."
	end
	table.insert(rTable, 'GAME_NAME')
	table.insert(rTable, game_name)
	-- SCORE
	local score = memorySum(0, 100000, 0x0240, 5, .1)
	table.insert(rTable, 'SCORE')
	table.insert(rTable, score)
	-- LIVES
	local lives = 0
	-- PLAYER X
	local px = 0
	-- PLAYER Y
	local pxy = 0
	return rTable
end

-- Initialize Variables
controllerPort = 1

-- Main Loop
while true do
	-- Read input data from file, and apply it
	input = readFile('input.txt')
	joypad.set(controllerPort, input)
	
	-- Read the screen's pixels and write to file
	writeFile('screen.txt', getScreenTable(), 4)
	writeFile('variables.txt', getMemoryValues(), 2)
	
	-- Advance the frame
	emu.frameadvance()
end