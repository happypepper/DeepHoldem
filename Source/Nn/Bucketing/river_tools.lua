local arguments = require 'Settings.arguments'

local M = {}

function M:_suitcat_river(s1,s2,s3,s4,s5,s6,s7)
	local suit = {}
  suit[0] = 0
  suit[1] = 0
  suit[2] = 0
  suit[3] = 0
	suit[s3] = suit[s3] + 1;
	suit[s4] = suit[s4] + 1;
	suit[s5] = suit[s5] + 1;
	suit[s6] = suit[s6] + 1;
	suit[s7] = suit[s7] + 1;

	if suit[0]<=2 and suit[1]<=2 and suit[2]<=2 and suit[3]<=2 then
		return 0
	end

	if suit[0]==3 or suit[1]==3 or suit[2]==3 or suit[3]==3 then
		local thesuit = -1;
		for i=0,3 do
			if suit[i]==3 then
				thesuit = i
      end
    end

		local mask = 0
		if s3==thesuit then mask = mask + 1 end
		if s4==thesuit then mask = mask + 2 end
		if s5==thesuit then mask = mask + 4 end
		if s6==thesuit then mask = mask + 8 end
		if s7==thesuit then mask = mask + 16 end

		local add = 0
		if s1==thesuit and s2==thesuit then add = 1
    elseif s1==thesuit then add = 2
		elseif s2==thesuit then add = 3 end

		if mask==7 then
			return 1 + add
		elseif mask==11 then
			return 5 + add
		elseif mask==19 then
			return 9 + add
		elseif mask==13 then
			return 13 + add
		elseif mask==21 then
			return 17 + add
		elseif mask==25 then
			return 21 + add
		elseif mask==14 then
			return 25 + add
		elseif mask==22 then
			return 29 + add
		elseif mask==26 then
			return 33 + add
		elseif mask==28 then
			return 37 + add
		end
		print("bad river suits")
		io.read()
	end

	if suit[0]==4 or suit[1]==4 or suit[2]==4 or suit[3]==4 then
		local thesuit = -1;
    for i=0,3 do
			if suit[i]==4 then
				thesuit = i
      end
    end

		if s3 ~= thesuit then
			if s1==thesuit and s2==thesuit then return 42 end
			if s1==thesuit then return 43 end
			if s2==thesuit then return 44 end
			return 45
		elseif s4~=thesuit then
			if s1==thesuit and s2==thesuit then return 46 end
			if s1==thesuit then return 47 end
			if s2==thesuit then return 48 end
			return 49
		elseif s5~=thesuit then
			if s1==thesuit and s2==thesuit then return 50 end
			if s1==thesuit then return 51 end
			if s2==thesuit then return 52 end
			return 53
		elseif s6~=thesuit then
			if s1==thesuit and s2==thesuit then return 54 end
			if s1==thesuit then return 55 end
			if s2==thesuit then return 56 end
			return 57
		elseif s7~=thesuit then
			if s1==thesuit and s2==thesuit then return 58 end
			if s1==thesuit then return 59 end
			if s2==thesuit then return 60 end
			return 61;
		end

		print("bad river suits")
		io.read()
	end
	if suit[0]==5 or suit[1]==5 or suit[2]==5 or suit[3]==5 then
    local thesuit = -1;
    for i=0,3 do
			if suit[i]==5 then
				thesuit = i
      end
    end

		if s1==thesuit and s2==thesuit then return 62 end
		if s1==thesuit then return 63 end
		if s2==thesuit then return 64 end
		return 65
	end

	print("bad river suits")
	io.read()
end

function M:riverID(h, b)

  local hole = h
  local board = torch.sort(b)
	local base =
         math.floor((hole[1]-1)/4)*13*13*13*13*13*13 +
				 math.floor((hole[2]-1)/4)*13*13*13*13*13 +
				 math.floor((board[1]-1)/4)*13*13*13*13 +
				 math.floor((board[2]-1)/4)*13*13*13 +
         math.floor((board[3]-1)/4)*13*13 +
         math.floor((board[4]-1)/4)*13 +
         math.floor((board[5]-1)/4);

	local suitcode = self:_suitcat_river(
    hole[1]%4,
    hole[2]%4,
    board[1]%4,
    board[2]%4,
    board[3]%4,
    board[4]%4,
    board[5]%4);
	if suitcode==-1 then
		print("error suit cat")
    io.read()
	end
	suitcode = suitcode * 815730722
	return suitcode+base
end

return M
