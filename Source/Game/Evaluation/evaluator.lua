--- Evaluates hand strength in Leduc Hold'em and variants.
--
-- Works with hands which contain two or three cards, but assumes that
-- the deck contains no more than two cards of each rank (so three-of-a-kind
-- is not a possible hand).
--
-- Hand strength is given as a numerical value, where a lower strength means
-- a stronger hand: high pair < low pair < high card < low card
-- @module evaluator

require 'torch'
require 'math'
local game_settings = require 'Settings.game_settings'
local card_to_string = require 'Game.card_to_string_conversion'
local card_tools = require 'Game.card_tools'
local arguments = require 'Settings.arguments'

local M = {_texas_lookup = nil}

function M:_init()
  self._idx_to_cards = arguments.Tensor(game_settings.hand_count, game_settings.hand_card_count)
  for card1 = 1, game_settings.card_count do
    for card2 = card1 + 1, game_settings.card_count do
      local idx = card_tools:get_hole_index({card1,card2})
      --print(card_to_string:card_to_string(card1) .. card_to_string:card_to_string(card2) .. ': ' .. idx)
      self._idx_to_cards[idx][1] = card1
      self._idx_to_cards[idx][2] = card2
    end
  end
  if self._texas_lookup == nil then
    local f = assert(io.open("./Game/Evaluation/HandRanks.dat", "rb"))
    local data = f:read("*all")
    self._texas_lookup = arguments.Tensor(string.len(data) / 4):fill(0):long()
    if arguments.gpu then
      self._texas_lookup = self._texas_lookup:cudaLong()
    end
    for i = 1, string.len(data), 4 do
      local num = 0
      for j = i,i+3 do
        num = num + data:byte(j) * (2 ^ ((j - i) * 8))
      end
      self._texas_lookup[(i - 1) / 4 + 1] = num
    end
    f:close()
  end
end

M:_init()

--- Gives a strength representation for a hand containing two cards.
-- @param hand_ranks the rank of each card in the hand
-- @return the strength value of the hand
-- @local
function M:evaluate_two_card_hand(hand_ranks)
  --check for the pair
  local hand_value = nil
  if hand_ranks[1] == hand_ranks[2] then
    --hand is a pair
    hand_value = hand_ranks[1]
  else
    --hand is a high card
    hand_value = hand_ranks[1] * game_settings.rank_count + hand_ranks[2]
  end
  return hand_value
end

--- Gives a strength representation for a hand containing three cards.
-- @param hand_ranks the rank of each card in the hand
-- @return the strength value of the hand
-- @local
function M:evaluate_three_card_hand(hand_ranks)
  local hand_value = nil
  --check for the pair
  if hand_ranks[1] == hand_ranks[2] then
    --paired hand, value of the pair goes first, value of the kicker goes second
    hand_value = hand_ranks[1] * game_settings.rank_count + hand_ranks[3]
  elseif hand_ranks[2] == hand_ranks[3] then
    --paired hand, value of the pair goes first, value of the kicker goes second
    hand_value = hand_ranks[2] * game_settings.rank_count + hand_ranks[1]
  else
    --hand is a high card
    hand_value = hand_ranks[1] * game_settings.rank_count * game_settings.rank_count + hand_ranks[2] * game_settings.rank_count + hand_ranks[3]
  end
  return hand_value
end

--- Gives a strength representation for a texas hold'em hand containing seven cards.
-- @param hand_ranks the rank of each card in the hand
-- @return the strength value of the hand
-- @local
function M:evaluate_seven_card_hand(hand)
  local rank = self._texas_lookup[54 + (hand[1] - 1) + 1]
  for c = 2, hand:size(1) do
    rank = self._texas_lookup[1 + rank + (hand[c] - 1) + 1]
  end
  return -rank
end

--- Gives a strength representation for a two or three card hand.
-- @param hand a vector of two or three cards
-- @param[opt] impossible_hand_value the value to return if the hand is invalid
-- @return the strength value of the hand, or `impossible_hand_value` if the
-- hand is invalid
function M:evaluate(hand, impossible_hand_value)
  assert(hand:max() <= game_settings.card_count and hand:min() > 0, 'hand does not correspond to any cards' )
  impossible_hand_value = impossible_hand_value or -1
  if not card_tools:hand_is_possible(hand) then
    return impossible_hand_value
  end
  --we are not interested in the hand suit - we will use ranks instead of cards
  if hand:size(1) == 2 then
    local hand_ranks = hand:clone()
    for i = 1, hand_ranks:size(1) do
      hand_ranks[i] = card_to_string:card_to_rank(hand_ranks[i])
    end
    hand_ranks = hand_ranks:sort()
    return self:evaluate_two_card_hand(hand_ranks)
  elseif hand:size(1) == 3 then
    local hand_ranks = hand:clone()
    for i = 1, hand_ranks:size(1) do
      hand_ranks[i] = card_to_string:card_to_rank(hand_ranks[i])
    end
    hand_ranks = hand_ranks:sort()
    return self:evaluate_three_card_hand(hand_ranks)
  elseif hand:size(1) == 7 then
    return self:evaluate_seven_card_hand(hand)
  else
    assert(false, 'unsupported size of hand!' )
  end
end

function M:evaluate_fast(hands)
  local ret = self._texas_lookup:index(1,torch.add(hands[{{},1}],54))
  for c = 2, hands:size(2) do
    ret = self._texas_lookup:index(1, torch.add(hands[{{},c}],ret):add(1))
  end
  ret:cmul(card_tools:get_possible_hands_mask(hands))
  ret:mul(-1)
  return ret
end

--- Gives strength representations for all private hands on the given board.
-- @param board a possibly empty vector of board cards
-- @param impossible_hand_value the value to assign to hands which are invalid
-- on the board
-- @return a vector containing a strength value or `impossible_hand_value` for
-- every private hand
function M:batch_eval(board, impossible_hand_value)
  local hand_values = arguments.Tensor(game_settings.hand_count):fill(-1)
  if board:dim() == 0 then -- kuhn poker
    for hand = 1, game_settings.card_count do
      hand_values[hand] = math.floor((hand - 1 ) / game_settings.suit_count ) + 1
    end
  else
    local board_size = board:size(1)
    assert(board_size == 1 or board_size == 2 or board_size == 5, 'Incorrect board size for Leduc' )

    local whole_hand = arguments.Tensor(board_size + game_settings.hand_card_count)
    whole_hand[{{1, -1 - game_settings.hand_card_count}}]:copy(board)

    if game_settings.hand_card_count == 1 then
      for card = 1, game_settings.card_count do
        whole_hand[-1] = card
        hand_values[card] = self:evaluate(whole_hand, impossible_hand_value)
      end
    elseif game_settings.hand_card_count == 2 then
      for card1 = 1, game_settings.card_count do
        for card2 = card1 + 1, game_settings.card_count do
          whole_hand[-2] = card1
          whole_hand[-1] = card2
          local idx = card_tools:get_hole_index({card1,card2})
          --print(card_to_string:card_to_string(card1) .. card_to_string:card_to_string(card2) .. ': ' .. idx)
          hand_values[idx] = self:evaluate(whole_hand, impossible_hand_value)
        end
      end
    else
      assert(false, "unsupported hand_card_count: " .. game_settings.hand_card_count)
    end
  end
  return hand_values
end

function M:batch_eval_fast(board)
  if board:dim() == 0 then -- kuhn poker
    return nil
  elseif board:dim() == 2 then
    local batch_size = board:size(1)
    local hands = arguments.Tensor(batch_size, game_settings.hand_count, board:size(2) + game_settings.hand_card_count):long()
    if arguments.gpu then
      hands = hands:cudaLong()
    end
    hands[{{},{},{1,board:size(2)}}]:copy(
      board:view(batch_size, 1, board:size(2))
           :expand(batch_size, game_settings.hand_count, board:size(2)))
    hands[{{},{},{-2,-1}}]:copy(
      self._idx_to_cards:view(1, game_settings.hand_count, game_settings.hand_card_count)
                        :expand(batch_size, game_settings.hand_count, game_settings.hand_card_count))
    return self:evaluate_fast(hands:view(-1, board:size(2) + game_settings.hand_card_count)):view(batch_size, game_settings.hand_count)
  elseif board:dim() == 1 then
    local hands = arguments.Tensor(game_settings.hand_count, board:size(1) + game_settings.hand_card_count):long()
    if arguments.gpu then
      hands = hands:cudaLong()
    end
    hands[{{},{1,board:size(1)}}]:copy(board:view(1,board:size(1)):expand(game_settings.hand_count,board:size(1)))
    hands[{{},{-2,-1}}]:copy(self._idx_to_cards)
    return self:evaluate_fast(hands)
  else
    assert(false, "weird board dim " .. board:dim())
  end
end

return M
