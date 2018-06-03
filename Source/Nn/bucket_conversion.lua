--- Converts between vectors over private hands and vectors over buckets.
-- @classmod bucket_conversion

require 'torch'
require 'math'
local card_tools = require 'Game.card_tools'
local arguments = require 'Settings.arguments'
local game_settings = require 'Settings.game_settings'
local bucketer = require 'Nn.bucketer'
local tools = require 'tools'

local BucketConversion = torch.class('BucketConversion')

--- Constructor
function BucketConversion:__init()
end

--- Sets the board cards for the bucketer.
-- @param board a non-empty vector of board cards
function BucketConversion:set_board(board, raw)
  if raw ~= nil then
    self.bucket_count = tools:choose(14, 2) + tools:choose(10, 2)
  else
    self.bucket_count = bucketer:get_bucket_count(card_tools:board_to_street(board))
  end
  self._range_matrix = arguments.Tensor(game_settings.hand_count, self.bucket_count ):zero()

  local buckets = nil
  if raw ~= nil then
    buckets = bucketer:compute_rank_buckets(board)
  else
    buckets = bucketer:compute_buckets(board)
  end
  local class_ids = torch.range(1, self.bucket_count)

  if arguments.gpu then
    buckets = buckets:cuda()
    class_ids = class_ids:cuda()
  else
    class_ids = class_ids:float()
  end

  class_ids = class_ids:view(1, self.bucket_count):expand(game_settings.hand_count, self.bucket_count)
  local card_buckets = buckets:view(game_settings.hand_count, 1):expand(game_settings.hand_count, self.bucket_count)

  --finding all strength classes
  --matrix for transformation from card ranges to strength class ranges
  self._range_matrix[torch.eq(class_ids, card_buckets)] = 1

  --matrix for transformation form class values to card values
  self._reverse_value_matrix = self._range_matrix:t():clone()
end

--- Converts a range vector over private hands to a range vector over buckets.
--
-- @{set_board} must be called first. Used to create inputs to the neural net.
-- @param card_range a probability vector over private hands
-- @param bucket_range a vector in which to save the resulting probability
-- vector over buckets
function BucketConversion:card_range_to_bucket_range(card_range, bucket_range)
  bucket_range:mm(card_range, self._range_matrix)
end

function BucketConversion:hand_cfvs_to_bucket_cfvs(card_range, card_cfvs, bucket_range, bucketed_cfvs)
  bucketed_cfvs:mm(torch.cmul(card_range,card_cfvs), self._range_matrix)

  -- avoid divide by 0
  bucketed_cfvs:cdiv(torch.cmax(bucket_range, 0.00001))
end

--- Converts a value vector over buckets to a value vector over private hands.
--
-- @{set_board} must be called first. Used to process neural net outputs.
-- @param bucket_value a vector of values over buckets
-- @param card_value a vector in which to save the resulting vector of values
-- over private hands
function BucketConversion:bucket_value_to_card_value(bucket_value, card_value)
  card_value:mm(bucket_value, self._reverse_value_matrix)
end

--- Gives a vector of possible buckets on the the board.
--
-- @{set_board} must be called first.
-- @return a mask vector over buckets where each entry is 1 if the bucket is
-- valid, 0 if not
function BucketConversion:get_possible_bucket_mask()
  local mask = arguments.Tensor(1, self.bucket_count)
  local card_indicator = arguments.Tensor(1, game_settings.hand_count):fill(1)

  mask:mm(card_indicator, self._range_matrix)
  mask[torch.gt(mask, 0)] = 1

  return mask
end
