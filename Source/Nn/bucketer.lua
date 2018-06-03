--- Assigns hands to buckets on the given board.
--
-- For the Leduc implementation, we simply assign every possible set of
-- private and board cards to a unique bucket.
-- @classmod bucketer
local game_settings = require 'Settings.game_settings'
local card_tools = require 'Game.card_tools'
local card_to_string = require 'Game.card_to_string_conversion'
local arguments = require 'Settings.arguments'
local constants = require 'Settings.constants'
local river_tools = require 'Nn.Bucketing.river_tools'
local turn_tools = require 'Nn.Bucketing.turn_tools'
local flop_tools = require 'Nn.Bucketing.flop_tools'
local card_to_string_conversion = require 'Game.card_to_string_conversion'
local evaluator = require 'Game.Evaluation.evaluator'
local tools = require 'tools'

local M = {}

function M:_init()
  if self._ihr_pair_to_bucket == nil then
    local f = assert(io.open("./Nn/Bucketing/riverihr.dat", "rb"))
    local data = f:read("*all")

    self._river_ihr = {}
    for i = 1, string.len(data), 7 do
      local key = 0
      for j = i,i+4 do
        key = key + data:byte(j) * (2 ^ ((4 - j + i) * 8))
      end
      local win = data:byte(i+5)
      local tie = data:byte(i+6)
      self._river_ihr[key] = win*200 + tie
    end
    f:close()

    local f = assert(io.open("./Nn/Bucketing/rcats.dat", "r"))
    self.river_buckets = f:read("*number")
    self._ihr_pair_to_bucket = {}
    for i = 1, self.river_buckets do
      local win = f:read("*number")
      local tie = f:read("*number")
      self._ihr_pair_to_bucket[win * 1000 + tie] = i
    end
    f:close()
  end

  if self._turn_means == nil then
    self._turn_means = {}
    local f = assert(io.open("./Nn/Bucketing/turn_means.dat"))
    local num_means = f:read("*number")
    for i = 1,num_means do
      local dist = {}
      for j = 0,50 do
        dist[j] = f:read("*number")
      end
      self._turn_means[i] = dist
    end
    f:close()
  end

  if self._turn_cats == nil then
    self._turn_cats = {}
    local f = assert(io.open("./Nn/Bucketing/turn_dist_cats.dat", "rb"))
    local data = f:read("*all")

    for i = 1, string.len(data), 6 do
      local key = 0
      for j = i,i+3 do
        key = key + data:byte(j) * (2 ^ ((j - i) * 8))
      end
      local cat = data:byte(i+4) + data:byte(i+5) * (2 ^ 8)
      self._turn_cats[key] = cat

      assert(cat <= 1000 and cat >= 1, "cat = " .. cat)
    end
    f:close()
  end
  if self._flop_cats == nil then
    self._flop_cats = {}
    local f = assert(io.open("./Nn/Bucketing/flop_dist_cats.dat", "rb"))
    local data = f:read("*all")

    for i = 1, string.len(data), 6 do
      local key = 0
      for j = i,i+3 do
        key = key + data:byte(j) * (2 ^ ((j - i) * 8))
      end
      local cat = data:byte(i+4) + data:byte(i+5) * (2 ^ 8)
      self._flop_cats[key] = cat

      assert(cat <= 1000 and cat >= 1, "cat = " .. cat)
    end
    f:close()
  end
end

M:_init()

--- Gives the total number of buckets across all boards.
-- @return the number of buckets
function M:get_bucket_count(street)
  if street == 4 then
    return self.river_buckets
  elseif street == 3 or street == 2 then
    return 1000
  elseif street == 1 then
    return 169
  end
  return 169
end

--- Gives the maximum number of ranks across all boards.
-- @return the number of buckets
function M:get_rank_count()
  return tools:choose(14, 2) + tools:choose(10, 2)
end

function M:emd(a,b)
  local emds = {}
  emds[0] = 0;
  for i = 1, 51 do
    emds[i] = a[i-1] + emds[i-1] - b[i-1];
  end
  local ret = 0;
  for i = 0, 51 do
    ret = ret + math.abs(emds[i]);
  end
  return ret
end

function M:_compute_turn_buckets(board)
  local buckets = torch.Tensor(game_settings.hand_count):fill(-1)
  local used = torch.ByteTensor(game_settings.card_count):fill(0)
  local hand = torch.ByteTensor(7)
  for i = 1, board:size(1) do
    used[board[i]] = 1
    hand[i + 2] = board[i]
  end

  for card1 = 1,game_settings.card_count do
    if used[card1] == 0 then
      used[card1] = 1
      hand[1] = card1
      for card2 = card1+1, game_settings.card_count do
        if used[card2] == 0 then
          used[card2] = 1
          hand[2] = card2

          local idx = card_tools:get_hole_index({card1,card2})

          --print(card_to_string_conversion:cards_to_string(hand[{{1,6}}]))
          local turn_code = turn_tools:turnID(hand[{{1,2}}]:clone(), hand[{{3,6}}]:clone())

          local closest_mean = self._turn_cats[turn_code]
          buckets[idx] = closest_mean

          used[card2] = 0
        end
      end
      used[card1] = 0
    end
  end

  return buckets
end

function M:_compute_flop_buckets(board)
  local buckets = torch.Tensor(game_settings.hand_count):fill(-1)
  local used = torch.ByteTensor(game_settings.card_count):fill(0)
  local hand = torch.ByteTensor(5)
  for i = 1, board:size(1) do
    used[board[i]] = 1
    hand[i + 2] = board[i]
  end

  for card1 = 1,game_settings.card_count do
    if used[card1] == 0 then
      used[card1] = 1
      hand[1] = card1
      for card2 = card1+1, game_settings.card_count do
        if used[card2] == 0 then
          used[card2] = 1
          hand[2] = card2

          local idx = card_tools:get_hole_index({card1,card2})

          local flop_code = flop_tools:flopID(hand[{{1,2}}]:clone(), hand[{{3,5}}]:clone())

          local closest_mean = self._flop_cats[flop_code]
          buckets[idx] = closest_mean

          used[card2] = 0
        end
      end
      used[card1] = 0
    end
  end
  return buckets
end

function M:_compute_preflop_buckets()
  if self._preflop_buckets == nil then
    self._preflop_buckets = arguments.Tensor(game_settings.hand_count):fill(-1)

    for card1 = 1,game_settings.card_count do
      for card2 = card1+1, game_settings.card_count do

        local idx = card_tools:get_hole_index({card1,card2})

        local rank1 = math.floor((card1 - 1) / 4)
        local rank2 = math.floor((card2 - 1) / 4)
        if card1 % 4 == card2 % 4 then
          self._preflop_buckets[idx] = rank1 * 13 + rank2 + 1
        else
          self._preflop_buckets[idx] = rank2 * 13 + rank1 + 1
        end
      end
    end
  end
  return self._preflop_buckets
end

function M:_compute_river_buckets(board)
  local buckets = torch.Tensor(game_settings.hand_count):fill(-1)
  local used = torch.ByteTensor(game_settings.card_count):fill(0)
  local board_size = board:size(1)
  for i = 1, board_size do
    used[board[i]] = 1
  end
  local hands = torch.ByteTensor(constants.players_count, board_size + game_settings.hand_card_count)
  for i = 1, constants.players_count do
    hands[{i,{1, - 1 - game_settings.hand_card_count}}]:copy(board)
  end

  for card1 = 1,game_settings.card_count do
    if used[card1] == 0 then
      used[card1] = 1
      hands[1][-2] = card1
      for card2 = card1+1, game_settings.card_count do
        if used[card2] == 0 then
          used[card2] = 1
          hands[1][-1] = card2
          local idx = card_tools:get_hole_index({card1,card2})

          local code = river_tools:riverID(hands[1][{{6,7}}], hands[1][{{1,5}}])
          local ihr = self._river_ihr[code]
          local win_bucket = math.floor(ihr/200)
          local tie_bucket = math.floor((ihr % 200)/2)

          local r_bucket = self._ihr_pair_to_bucket[win_bucket * 1000 + tie_bucket]
          assert(r_bucket ~= nil, 'bad win,tie ihr pair')
          buckets[idx] = r_bucket
          used[card2] = 0
        end
      end
      used[card1] = 0
    end
  end

  return buckets
end

--- Gives a vector which maps private hands to buckets on a given board.
-- @param board a non-empty vector of board cards
-- @return a vector which maps each private hand to a bucket index
function M:compute_buckets(board)
  local street = card_tools:board_to_street(board)

  if street == 4 then
    return self:_compute_river_buckets(board)
  elseif street == 3 then
    return self:_compute_turn_buckets(board)
  elseif street == 2 then
    return self:_compute_flop_buckets(board)
  elseif street == 1 then
    return self:_compute_preflop_buckets()
  end
end

function M:compute_rank_buckets(board)
  local buckets = arguments.Tensor(game_settings.hand_count):fill(-1)

  local ranks = evaluator:batch_eval(board,-1)
  local sorted_ranks, _ = torch.sort(ranks)
  local rank_idx = 0
  local rank_idxs = {}
  for i = 1, sorted_ranks:size(1) do
    if sorted_ranks[i] == -1 then
      break
    end
    if (i > 1 and sorted_ranks[i] ~= sorted_ranks[i-1]) or i == 1 then
      rank_idx = rank_idx + 1
      rank_idxs[sorted_ranks[i]] = rank_idx
    end
  end

  for i = 1, ranks:size(1) do
    if ranks[i] ~= -1 then
      ranks[i] = rank_idxs[ranks[i]]
    end
  end

  return ranks
end

return M
