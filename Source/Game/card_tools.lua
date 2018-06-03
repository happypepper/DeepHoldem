--- A set of tools for basic operations on cards and sets of cards.
--
-- Several of the functions deal with "range vectors", which are probability
-- vectors over the set of possible private hands. For Leduc Hold'em,
-- each private hand consists of one card.
-- @module card_tools
local game_settings = require 'Settings.game_settings'
local arguments = require 'Settings.arguments'
local constants = require 'Settings.constants'
local tools = require 'tools'
local card_to_string_conversion = require 'Game.card_to_string_conversion'

local M = {}

--- Gives whether a set of cards is valid.
-- @param hand a vector of cards
-- @return `true` if the tensor contains valid cards and no card is repeated
function M:hand_is_possible(hand)
  assert(hand:min() > 0 and hand:max() <= game_settings.card_count, 'Illegal cards in hand' )
  local used_cards = torch.FloatTensor(game_settings.card_count):fill(0);
  for i = 1, hand:size(1) do
    used_cards[hand[i]] = used_cards[hand[i]] + 1
  end
  return used_cards:max() < 2
end

function M:get_possible_hands_mask(hands)
  local used_cards = arguments.Tensor(hands:size(1), game_settings.card_count):fill(0)

  used_cards:scatterAdd(2,hands,arguments.Tensor(hands:size(1), 7):fill(1))
  local ret = torch.le(torch.max(used_cards, 2), 1):long()
  if arguments.gpu then
    ret = ret:cudaLong()
  end

  return ret
end

--- Gives the private hands which are valid with a given board.
-- @param board a possibly empty vector of board cards
-- @return a vector with an entry for every possible hand (private card), which
--  is `1` if the hand shares no cards with the board and `0` otherwise
-- TODO generalize
function M:get_possible_hand_indexes(board)
  local out = arguments.Tensor(game_settings.hand_count):fill(0)
  if board:dim() == 0 then
    out:fill(1)
    return out
  end

  local used = {}
  for i = 1, board:size(1) do
    used[board[i]] = 1
  end

  for card1 = 1, game_settings.card_count do
    if not used[card1] then
      for card2 = card1+1, game_settings.card_count do
        if not used[card2] then
          out[M:get_hole_index({card1,card2})] = 1
        end
      end
    end
  end
  return out
end

--- Gives the private hands which are invalid with a given board.
-- @param board a possibly empty vector of board cards
-- @return a vector with an entry for every possible hand (private card), which
-- is `1` if the hand shares at least one card with the board and `0` otherwise
function M:get_impossible_hand_indexes(board)
  local out = self:get_possible_hand_indexes(board)
  out:add(-1)
  out:mul(-1)
  return out
end

--- Gives a range vector that has uniform probability on each hand which is
-- valid with a given board.
-- @param board a possibly empty vector of board cards
-- @return a range vector where invalid hands have 0 probability and valid
-- hands have uniform probability
function M:get_uniform_range(board)
  local out = self:get_possible_hand_indexes(board)
  out:div(out:sum())

  return out
end

function M:get_file_range(filename)
  local out = arguments.Tensor(game_settings.hand_count):fill(0)
  local f = assert(io.open(filename, "r"))
  while true do
    local s = f:read(4)
    if s == nil then
      break
    end
    s = s:gsub("t","T")
    s = s:gsub("j","J")
    s = s:gsub("q","Q")
    s = s:gsub("k","K")
    s = s:gsub("a","A")
    local n = f:read("*number")
    if n == nil then
      break
    end
    -- read newline
    f:read(1)

    local hand = card_to_string_conversion:string_to_board(s)
    if hand[1] > hand[2] then
      local temp = hand[1]
      hand[1] = hand[2]
      hand[2] = temp
    end
    local idx = self:get_hole_index({hand[1], hand[2]})
    out[idx] = n
  end
  out:div(out:sum())
  return out
end

--- Randomly samples a range vector which is valid with a given board.
-- @param board a possibly empty vector of board cards
-- @param[opt] seed a seed for the random number generator
-- @return a range vector where invalid hands are given 0 probability, each
-- valid hand is given a probability randomly sampled from the uniform
-- distribution on [0,1), and the resulting range is normalized
function M:get_random_range(board, seed)
  seed = seed or torch.random()

  local gen = torch.Generator()
  torch.manualSeed(gen, seed)

  local out = torch.rand(gen, game_settings.hand_count):typeAs(arguments.Tensor())
  out:cmul(self:get_possible_hand_indexes(board))
  out:div(out:sum())

  return out
end

--- Checks if a range vector is valid with a given board.
-- @param range a range vector to check
-- @param board a possibly empty vector of board cards
-- @return `true` if the range puts 0 probability on invalid hands and has
-- total probability 1
function M:is_valid_range(range, board)
  local check = range:clone()
  local only_possible_hands = range:clone():cmul(self:get_impossible_hand_indexes(board)):sum() == 0
  local sums_to_one = math.abs(1.0 - range:sum()) < 0.0001
  return only_possible_hands and sums_to_one
end

--- Gives the current betting round based on a board vector.
-- @param board a possibly empty vector of board cards
-- @return the current betting round
function M:board_to_street(board)
  if board:dim() == 0 then
    return 1
  else
    for i = 1,constants.streets_count do
      if board:size(1) == game_settings.board_card_count[i] then
        return i
      end
    end
    assert(false, 'bad board dims')
  end
end

function M:_build_boards(boards, cur_board, out, card_index, last_index, base_index)
  if card_index == last_index + 1 then
    for i = 1, last_index do
      boards.boards[boards.index][i] = cur_board[i]
    end
    out[boards.index]:copy(cur_board)
    boards.index = boards.index + 1
    return
  end

  local startindex = 1
  if card_index > base_index then
    startindex = cur_board[card_index-1] + 1
  end
  for i = startindex, game_settings.card_count do
    local good = true
    for j = 1, card_index - 1 do
      if cur_board[j] == i then
        good = false
      end
    end
    if good then
      cur_board[card_index] = i
      self:_build_boards(boards,cur_board, out, card_index+1, last_index, base_index)
    end
  end
end

--- Gives all possible sets of board cards for the game.
-- @return an NxK tensor, where N is the number of possible boards, and K is
-- the number of cards on each board
function M:get_next_round_boards(board)
  local street = self:board_to_street(board)
  local boards_count = self:get_next_boards_count(street)
  local out = arguments.Tensor(boards_count, game_settings.board_card_count[street+1])
  local boards = {index = 1, boards = out}
  local cur_board = arguments.Tensor(game_settings.board_card_count[street+1])
  if board:dim() > 0 then
    for i = 1, board:size(1) do
      cur_board[i] = board[i]
    end
  end

  self:_build_boards(boards, cur_board, out,
    game_settings.board_card_count[street] + 1,
    game_settings.board_card_count[street+1],
    game_settings.board_card_count[street] + 1)
--  assert(boards.index == boards_count, boards.index .. ' ' .. boards_count)
  if self.flop_board_idx == nil and board:dim() == 0 then
    self.flop_board_idx = arguments.Tensor(game_settings.card_count, game_settings.card_count, game_settings.card_count)
    for i = 1, boards_count do
      local card1 = out[i][1]
      local card2 = out[i][2]
      local card3 = out[i][3]
      self.flop_board_idx[card1][card2][card3] = i
      self.flop_board_idx[card1][card3][card2] = i
      self.flop_board_idx[card2][card1][card3] = i
      self.flop_board_idx[card2][card3][card1] = i
      self.flop_board_idx[card3][card1][card2] = i
      self.flop_board_idx[card3][card2][card1] = i
    end
  end
  return out
end

--- Gives all possible sets of board cards for the game.
-- @return an NxK tensor, where N is the number of possible boards, and K is
-- the number of cards on each board
function M:get_last_round_boards(board)
  local street = self:board_to_street(board)
  local boards_count = self:get_last_boards_count(street)
  local out = arguments.Tensor(boards_count, game_settings.board_card_count[constants.streets_count])
  local boards = {index = 1, boards = out}
  local cur_board = arguments.Tensor(game_settings.board_card_count[constants.streets_count])
  if board:dim() > 0 then
    for i = 1, board:size(1) do
      cur_board[i] = board[i]
    end
  end

  self:_build_boards(boards, cur_board, out,
    game_settings.board_card_count[street] + 1,
    game_settings.board_card_count[constants.streets_count],
    game_settings.board_card_count[street] + 1)
--  assert(boards.index == boards_count, boards.index .. ' ' .. boards_count)
  return out
end

--- Gives the number of possible boards.
-- @return the number of possible boards
function M:get_next_boards_count(street)
  local used_cards = game_settings.board_card_count[street]

  local new_cards = game_settings.board_card_count[street+1] - game_settings.board_card_count[street]
  return tools:choose(game_settings.card_count - used_cards, new_cards)
end


--- Gives the number of possible boards.
-- @return the number of possible boards
function M:get_last_boards_count(street)
  local used_cards = game_settings.board_card_count[street]

  local new_cards = game_settings.board_card_count[constants.streets_count] - game_settings.board_card_count[street]
  return tools:choose(game_settings.card_count - used_cards, new_cards)
end

--- Gives a numerical index for a set of board cards.
-- @param board a non-empty vector of board cards
-- @return the numerical index for the board
function M:get_board_index(board)
  assert(board:size(1) > 3)

  local used_cards = arguments.Tensor(game_settings.card_count):fill(0)
  for i = 1, board:size(1) - 1 do
    used_cards[board[i]] = 1
  end
  local ans = 0
  for i = 1, game_settings.card_count do
    if used_cards[i] == 0 then
      ans = ans + 1
    end
    if i == board[-1] then
      return ans
    end
  end
  return -1
end

--- Gives a numerical index for a set of board cards.
-- @param board a non-empty vector of board cards
-- @return the numerical index for the board
function M:get_flop_board_index(board)
  if self.flop_board_idx == nil then
    self:get_next_round_boards(arguments.Tensor())
  end
  return self.flop_board_idx[board[1]][board[2]][board[3]]
end

--- Gives a numerical index for a set of hole cards.
-- @param hand a non-empty vector of hole cards, sorted
-- @return the numerical index for the hand
function M:get_hole_index(hand)
  local index = 1
  for i = 1, #hand do
    index = index + tools:choose(hand[i] - 1, i)
  end
  return index
end

--- Gives a numerical index for a set of hole cards.
-- @param hand a non-empty vector of hole cards, sorted
-- @return the numerical index for the hand
function M:string_to_hole_index(hand_string)
  local hole = card_to_string_conversion:string_to_board(hand_string)
  hole = torch.sort(hole)
  index = 1
  for i = 1, hole:size(1) do
    index = index + tools:choose(hole[i] - 1, i)
  end
  return index
end

--- Normalizes a range vector over hands which are valid with a given board.
-- @param board a possibly empty vector of board cards
-- @param range a range vector
-- @return a modified version of `range` where each invalid hand is given 0
-- probability and the vector is normalized
function M:normalize_range(board, range)
  local mask = self:get_possible_hand_indexes(board)
  local out = range:clone():cmul(mask)
  --return zero range if it all collides with board (avoid div by zero)
  if out:sum() == 0 then
    return out
  end
  out:div(out:sum())
  return out
end


return M
