local M = {}

function M:_suitcat_turn(s1,s2,s3,s4,s5,s6)
	if s1~=0 then return -1 end

	local ret = -1

	if s2==0 then
		if s3==0 then
			if s4==0 then
			  ret = s5 * 2 + s6
			elseif s4==1 then
				ret = 5 + s5 * 3 + s6
			end
		elseif s3==1 then
			if s4==0 then
				ret = 15 + s5 * 3 + s6
			elseif s4==1 then
				ret = 25 + s5 * 3 + s6
			elseif s4==2 then
				ret = 35 + s5 * 4 + s6
			end
		end
	elseif s2==1 then
		if s3==0 then
			if s4==0 then
				ret = 51 + s5 * 3 + s6
			elseif s4==1 then
				ret = 61 + s5 * 3 + s6
			elseif s4==2 then
				ret = 71 + s5 * 4 + s6
			end
		elseif s3==1 then
			if s4==0 then
				ret = 87 + s5 * 3 + s6
			elseif s4==1 then
				ret = 97 + s5 * 3 + s6
			elseif s4==2 then
				ret = 107 + s5 * 4 + s6
			end
		elseif s3==2 then
			ret = 123 + s4 * 16 + s5 * 4 + s6
		end
	end
	return ret
end

function M:turnID(h, b)

	b = torch.sort(b)

	-- Get hand suits
	local os = {}
  for i = 0,5 do
		if i <= 1 then
			os[i] = (h[i+1]-1)%4
		else
			os[i] = (b[i-1]-1)%4
		end
	end

  -- Canonicalize suits
	local MM = 0
	local s = {}
	for i=0,5 do
		local j = 0
		while j < i do
			if os[j] == os[i] then
				s[i] = s[j]
				break
			end
			j = j + 1
		end
		if j == i then
			s[i] = MM
			MM = MM + 1
		end

		if i <= 1 then
			local suitdiff = s[i] - ((h[i+1] - 1) % 4)
			h[i+1] = h[i+1] + suitdiff
		else
			local suitdiff = s[i] - ((b[i-1] - 1) % 4)
			b[i-1] = b[i-1] + suitdiff
		end
	end

	b = torch.sort(b)
	local base =
         math.floor((h[1]-1)/4)*13*13*13*13*13 +
				 math.floor((h[2]-1)/4)*13*13*13*13 +
				 math.floor((b[1]-1)/4)*13*13*13 +
				 math.floor((b[2]-1)/4)*13*13 +
         math.floor((b[3]-1)/4)*13 +
         math.floor((b[4]-1)/4)

	for i = 0,5 do
 		if i <= 1 then
 			s[i] = (h[i+1]-1)%4
 		else
 			s[i] = (b[i-1]-1)%4
 		end
 	end

	local cat = self:_suitcat_turn(s[0],s[1],s[2],s[3],s[4],s[5])

	if cat==-1 then
		print("error suit cat")
    io.read()
	end
	cat = cat*13*13*13*13*13*13 + base;
	return cat
end

return M
