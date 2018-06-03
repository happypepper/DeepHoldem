--- Samples random probability vectors for use as player ranges.
-- @classmod range_generator

require "math"
require "torch"
local arguments = require 'Settings.arguments'
local game_settings = require 'Settings.game_settings'
local evaluator = require 'Game.Evaluation.evaluator'
local card_tools = require 'Game.card_tools'
local card_to_string = require 'Game.card_to_string_conversion'

local RangeGenerator = torch.class('RangeGenerator')

--- Recursively samples a section of the range vector.
-- @param cards an NxJ section of the range tensor, where N is the batch size
-- and J is the length of the range sub-vector
-- @param mass a vector of remaining probability mass for each batch member
-- @see generate_range
-- @local
function RangeGenerator:_generate_recursion(cards, mass)
  local batch_size = cards:size(1)
  assert(mass:size(1) == batch_size)
  --we terminate recursion at size of 1
  local card_count = cards:size(2)
  if card_count == 1 then
    cards:copy(mass)
  else
    local rand = torch.rand(batch_size)
    if arguments.gpu then
      rand = rand:cuda()
    end
    local mass1 = mass:clone():cmul(rand)
    mass1[torch.lt(mass1, 0.00001)] = 0
    mass1[torch.gt(mass1, 0.99999)] = 1
    local mass2 = mass -mass1
    local halfSize = card_count/2
    --if the tensor contains an odd number of cards, randomize which way the
	--middle card goes
    if halfSize % 1 ~= 0 then
      halfSize = halfSize - 0.5
      halfSize = halfSize + torch.random(0,1)
    end
    self:_generate_recursion(cards[{{}, {1, halfSize}}], mass1)
    self:_generate_recursion(cards[{{}, {halfSize +1, -1}}], mass2)
  end
end

--- Samples a batch of ranges with hands sorted by strength on the board.
-- @param range a NxK tensor in which to store the sampled ranges, where N is
-- the number of ranges to sample and K is the range size
-- @see generate_range
-- @local
function RangeGenerator:_generate_sorted_range(range)
  local batch_size = range:size(1)
  self:_generate_recursion(range, arguments.Tensor(batch_size):fill(1))
end

--- Sets the (possibly empty) board cards to sample ranges with.
--
-- The sampled ranges will assign 0 probability to any private hands that
-- share any cards with the board.
--
-- @param board a possibly empty vector of board cards
function RangeGenerator:set_board(te, board)

  local hand_strengths = arguments.Tensor(game_settings.hand_count)
  for i=1,game_settings.hand_count do
    hand_strengths[i] = i
  end
  if board:dim() == 0 then
    hand_strengths = te:get_hand_strengths():squeeze()
  elseif board:size(1) == 5 then
    hand_strengths = evaluator:batch_eval(board)
  else
    hand_strengths = te:get_hand_strengths():squeeze()
  end
  local possible_hand_indexes = card_tools:get_possible_hand_indexes(board)
  self.possible_hands_count = possible_hand_indexes:sum(1)[1]
  self.possible_hands_mask = possible_hand_indexes:view(1, -1)
  if not arguments.gpu then
    self.possible_hands_mask = self.possible_hands_mask:byte()
  else
    self.possible_hands_mask = self.possible_hands_mask:cudaByte()
  end
  local non_colliding_strengths = arguments.Tensor(self.possible_hands_count)
  non_colliding_strengths:maskedSelect(hand_strengths, self.possible_hands_mask)
  local order
  _, order = non_colliding_strengths:sort()
  _, self.reverse_order = order:sort()
  self.reverse_order = self.reverse_order:view(1, -1):long()
  self.reordered_range = arguments.Tensor()
  self.sorted_range =arguments.Tensor()
end

--- Samples a batch of random range vectors.
--
-- Each vector is sampled indepently by randomly splitting the probability
-- mass between the bottom half and the top half of the range, and then
-- recursing on the two halfs.
--
-- @{set_board} must be called first.
--
-- @param range a NxK tensor in which to store the sampled ranges, where N is
-- the number of ranges to sample and K is the range size
function RangeGenerator:generate_range(range)
  local batch_size = range:size(1)
  self.sorted_range:resize(batch_size, self.possible_hands_count)
  self:_generate_sorted_range(self.sorted_range, self.possible_hands_count)
  --we have to reorder the the range back to undo the sort by strength
  local index = self.reverse_order:expandAs(self.sorted_range)
  if arguments.gpu then
    index = index:cuda()
  end
  self.reordered_range = self.sorted_range:gather(2, index)

  range:zero()
  range:maskedCopy(self.possible_hands_mask:expandAs(range), self.reordered_range)

end
