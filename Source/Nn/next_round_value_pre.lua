--- Uses the neural net to estimate value at the end of the first betting round.
-- @classmod next_round_value

require 'torch'
require 'math'
local bucketer = require 'Nn.bucketer'
local card_tools = require 'Game.card_tools'
local arguments = require 'Settings.arguments'
local game_settings = require 'Settings.game_settings'
local constants = require 'Settings.constants'
local tools = require 'tools'

local NextRoundValuePre = torch.class('NextRoundValuePre')

--- Constructor.
--
-- Creates a tensor that can translate hand ranges to bucket ranges
-- on any board.
-- @param nn the neural network

function NextRoundValuePre:__init(nn, aux_nn, board)
  self.nn = nn
  self.aux_nn = aux_nn
  self:_init_bucketing(board)
end

--- Initializes the tensor that translates hand ranges to bucket ranges.
-- @local
function NextRoundValuePre:_init_bucketing(board)
  local street = card_tools:board_to_street(board)
  self._street = street
  self.bucket_count = bucketer:get_bucket_count(street+1)
  local boards = card_tools:get_next_round_boards(board)
  self.boards = boards

  self.board_count = boards:size(1)
  self.board_buckets = arguments.Tensor(self.board_count, game_settings.hand_count)

  for idx = 1, self.board_count do
    if idx % 100 == 0 then
      print(idx)
    end
    local board = self.boards[idx]
    self.board_buckets[{idx,{}}]:copy(bucketer:compute_buckets(board))
  end
  self.impossible_mask = torch.lt(self.board_buckets,0)
  self.board_indexes = self.board_buckets:clone()
  self.board_indexes:maskedFill(self.impossible_mask, 1)
  self.board_indexes_scatter = self.board_buckets:clone()
  self.board_indexes_scatter:maskedFill(self.impossible_mask, self.bucket_count+1)

  if arguments.gpu then
    self.board_indexes = self.board_indexes:cudaLong()
    self.board_indexes_scatter = self.board_indexes_scatter:cudaLong()
  else
    self.board_indexes = self.board_indexes:long()
    self.board_indexes_scatter = self.board_indexes_scatter:long()
  end

  -- compute aux variables
  self.bucket_count_aux = bucketer:get_bucket_count(street)
  local pf_buckets = bucketer:compute_buckets(arguments.Tensor({}))

  local class_ids = torch.range(1, self.bucket_count_aux)

  if arguments.gpu then
    class_ids = class_ids:cuda()
  else
    class_ids = class_ids:float()
  end
  class_ids = class_ids:view(1, self.bucket_count_aux):expand(game_settings.hand_count, self.bucket_count_aux)
  local card_buckets = pf_buckets:view(game_settings.hand_count, 1):expand(game_settings.hand_count, self.bucket_count_aux)

  self._range_matrix_aux = arguments.Tensor(game_settings.hand_count, self.bucket_count_aux):zero()
  self._range_matrix_aux[torch.eq(class_ids, card_buckets)] = 1
  self._reverse_value_matrix_aux = self._range_matrix_aux:t():clone()

  local num_new_cards = game_settings.board_card_count[2] - game_settings.board_card_count[1]
  local num_cur_cards = game_settings.board_card_count[1]

  local den = tools:choose(
    game_settings.card_count - num_cur_cards - 2*game_settings.hand_card_count,
    num_new_cards)
  self.weight_constant = 1/den
end

--- Converts a range vector over private hands to a range vector over buckets.
-- @param card_range a probability vector over private hands
-- @param bucket_range a vector in which to store the output probabilities
--  over buckets
-- @local
function NextRoundValuePre:_card_range_to_bucket_range(card_range, bucket_range)
  local other_bucket_range = bucket_range:view(-1,self.board_count,self.bucket_count + 1):zero()

  local indexes = self.board_indexes_scatter:view(1,self.board_count, game_settings.hand_count)
    :expand(bucket_range:size(1), self.board_count, game_settings.hand_count)
  other_bucket_range:scatterAdd(
    3,
    indexes,
    card_range
      :view(-1,1,game_settings.hand_count)
      :expand(card_range:size(1),self.board_count, game_settings.hand_count))
end

function NextRoundValuePre:_card_range_to_bucket_range_aux(card_range, bucket_range)
  bucket_range:mm(card_range, self._range_matrix_aux)
end

function NextRoundValuePre:_card_range_to_bucket_range_on_board(board_idx, card_range, bucket_range)
  local other_bucket_range = bucket_range:view(-1,self.bucket_count + 1):zero()

  local indexes = self.board_indexes_scatter:view(1,self.board_count, game_settings.hand_count)[{{},board_idx,{}}]
    :expand(bucket_range:size(1), game_settings.hand_count)
  other_bucket_range:scatterAdd(
    2,
    indexes,
    card_range
      :view(-1,game_settings.hand_count)
      :expand(card_range:size(1), game_settings.hand_count))
end

--- Converts a value vector over buckets to a value vector over private hands.
-- @param bucket_value a value vector over buckets
-- @param card_value a vector in which to store the output values over
-- private hands

-- @local
function NextRoundValuePre:_bucket_value_to_card_value(bucket_value, card_value)
  local indexes = self.board_indexes:view(1,self.board_count, game_settings.hand_count)
    :expand(bucket_value:size(1), self.board_count, game_settings.hand_count)

  self.values_per_board:gather(bucket_value:view(bucket_value:size(1), self.board_count, self.bucket_count), 3, indexes)
  local impossible = self.impossible_mask:view(1,self.board_count, game_settings.hand_count)
    :expand(bucket_value:size(1), self.board_count, game_settings.hand_count)
  self.values_per_board:maskedFill(impossible,0)
  card_value:sum(self.values_per_board,2)
  card_value:mul(self.weight_constant)
end

function NextRoundValuePre:_bucket_value_to_card_value_aux(bucket_value, card_value)
  card_value:mm(bucket_value, self._reverse_value_matrix_aux)
end

--- Converts a value vector over buckets to a value vector over private hands
-- given a particular set of board cards.
-- TODO: fix this
-- @param board a non-empty vector of board cards
-- @param bucket_value a value vector over buckets
-- @param card_value a vector in which to store the output values over
-- private hands
-- @local
function NextRoundValuePre:_bucket_value_to_card_value_on_board(board, bucket_value, card_value)
  local board_idx = card_tools:get_flop_board_index(board)
  local indexes = self.board_indexes:view(1,self.board_count, game_settings.hand_count)[{{},board_idx,{}}]
    :expand(bucket_value:size(1), game_settings.hand_count)

  self.values_per_board:gather(bucket_value:view(bucket_value:size(1), self.bucket_count), 2, indexes)
  local impossible = self.impossible_mask:view(1,self.board_count, game_settings.hand_count)[{{},board_idx,{}}]
    :expand(bucket_value:size(1), game_settings.hand_count)
  self.values_per_board:maskedFill(impossible,0)
  card_value:copy(self.values_per_board)
end

--- Initializes the value calculator with the pot size of each state that
-- we are going to evaluate.
--
-- During continual re-solving, there is one pot size for each initial state
-- of the second betting round (before board cards are dealt).
-- @param pot_sizes a vector of pot sizes
-- betting round ends
function NextRoundValuePre:start_computation(pot_sizes, batch_size)
  self.iter = 0
  if pot_sizes:dim() == 0 then
    return
  end
  self.pot_sizes = pot_sizes:view(-1, 1):clone()
  self.pot_sizes = self.pot_sizes:expand(self.pot_sizes:size(1),batch_size):clone()
  self.pot_sizes = self.pot_sizes:view(-1, 1)
  self.batch_size = self.pot_sizes:size(1)
end

function NextRoundValuePre:get_value_aux(ranges, values, next_board_idx)
  assert(ranges and values)
  assert(ranges:size(1) == self.batch_size, self.batch_size .. " " .. ranges:size(1))
  self.iter = self.iter + 1
  if self.iter == 1 then
    self.next_round_inputs = arguments.Tensor(self.batch_size, (self.bucket_count_aux * constants.players_count + 1)):zero()
    self.next_round_values = arguments.Tensor(self.batch_size, constants.players_count, self.bucket_count_aux):zero()
    self.next_round_extended_range = arguments.Tensor(self.batch_size, constants.players_count, self.bucket_count_aux):zero()
    self.next_round_serialized_range = self.next_round_extended_range:view(-1, self.bucket_count_aux)
    self.values_per_board = arguments.Tensor(self.batch_size * constants.players_count, game_settings.hand_count)
    self.range_normalization = arguments.Tensor()
    self.value_normalization = arguments.Tensor(self.batch_size, constants.players_count)

    local den = 0
    assert(self._street <= 3)

    if game_settings.nl then
      den = arguments.stack
    else
      if self._street == 4 then
        den = 48
      elseif self._street == 3 then
        den = 48
      elseif self._street == 2 then
        den = 24
      elseif self._street == 1 then
        den = 10
      else
        den = -1
      end
    end

    --handling pot feature for the nn
    local nn_bet_input = self.pot_sizes:clone():mul(1/den)
    self.next_round_inputs[{{}, {-1}}]:copy(nn_bet_input)
  end
  local use_memory = self.iter > arguments.cfr_skip_iters and next_board_idx ~= nil
  if use_memory and self.iter == arguments.cfr_skip_iters + 1 then
    --first iter that we need to remember something - we need to init data structures
    self.bucket_range_on_board = arguments.Tensor(self.batch_size * constants.players_count, self.bucket_count)
    self.range_normalization_on_board = arguments.Tensor()
    self.value_normalization_on_board = arguments.Tensor(self.batch_size, constants.players_count)
    self.range_normalization_memory = arguments.Tensor(self.batch_size * constants.players_count, 1):zero()
    self.counterfactual_value_memory = arguments.Tensor(self.batch_size, constants.players_count, self.bucket_count):zero()
    self.next_round_extended_range_on_board = arguments.Tensor(self.batch_size, constants.players_count, self.bucket_count + 1):zero()
    self.next_round_serialized_range_on_board = self.next_round_extended_range_on_board:view(-1, self.bucket_count + 1)
    self.next_round_inputs_on_board = arguments.Tensor(self.batch_size, (self.bucket_count * constants.players_count + 1)):zero()
    self.next_round_values_on_board = arguments.Tensor(self.batch_size, constants.players_count, self.bucket_count):zero()

    -- copy pot features over
    self.next_round_inputs_on_board[{{}, {-1}}]:copy(self.next_round_inputs[{{},{-1}}])
  end

  --computing bucket range in next street for both players at once
  self:_card_range_to_bucket_range_aux(
    ranges:view(self.batch_size * constants.players_count, -1),
    self.next_round_extended_range:view(self.batch_size * constants.players_count, -1))

  self.range_normalization:sum(self.next_round_serialized_range[{{},{1,self.bucket_count_aux}}], 2)
  local rn_view = self.range_normalization:view(self.batch_size, constants.players_count)
  for player = 1, constants.players_count do
    self.value_normalization[{{}, player}]:copy(rn_view[{{}, 3 - player}])
  end

  if use_memory then
    self:_card_range_to_bucket_range_on_board(
      next_board_idx,
      ranges:view(self.batch_size * constants.players_count, -1),
      self.next_round_extended_range_on_board:view(self.batch_size * constants.players_count, -1))

    self.range_normalization_on_board:sum(self.next_round_serialized_range_on_board[{{},{1,self.bucket_count}}], 2)
    local rnb_view = self.range_normalization_on_board:view(self.batch_size, constants.players_count)
    for player = 1, constants.players_count do
      self.value_normalization_on_board[{{}, player}]:copy(rnb_view[{{}, 3 - player}])
    end
    self.range_normalization_memory:add(self.value_normalization_on_board)
  end

  --eliminating division by zero
  self.range_normalization[torch.eq(self.range_normalization, 0)] = 1
  self.next_round_serialized_range:cdiv(self.range_normalization:expandAs(self.next_round_serialized_range))
  for player = 1, constants.players_count do
    local player_range_index = {(player -1) * self.bucket_count_aux + 1, player * self.bucket_count_aux}
    self.next_round_inputs[{{}, player_range_index}]:copy(self.next_round_extended_range[{{},player, {1, self.bucket_count_aux}}])
  end

  --using nn to compute values
  local serialized_inputs_view= self.next_round_inputs:view(self.batch_size, -1)
  local serialized_values_view= self.next_round_values:view(self.batch_size, -1)

  --computing value in the next round
  self.aux_nn:get_value(serialized_inputs_view, serialized_values_view)

  if use_memory then
    --eliminating division by zero
    self.range_normalization_on_board[torch.eq(self.range_normalization_on_board, 0)] = 1
    self.next_round_serialized_range_on_board:cdiv(self.range_normalization_on_board:expandAs(self.next_round_serialized_range_on_board))
    for player = 1, constants.players_count do
      local player_range_index = {(player -1) * self.bucket_count + 1, player * self.bucket_count}
      self.next_round_inputs_on_board[{{}, player_range_index}]:copy(self.next_round_extended_range_on_board[{{},player, {1, self.bucket_count}}])
    end

    --using nn to compute values
    local serialized_inputs_view_on_board = self.next_round_inputs_on_board:view(self.batch_size, -1)
    local serialized_values_view_on_board = self.next_round_values_on_board:view(self.batch_size, -1)

    --computing value in the next round
    self.nn:get_value(serialized_inputs_view_on_board, serialized_values_view_on_board)
  end

  --normalizing values back according to the orginal range sum
  local normalization_view = self.value_normalization:view(self.batch_size, constants.players_count, 1)
  self.next_round_values:cmul(normalization_view:expandAs(self.next_round_values))

  if use_memory then
    local normalization_view_on_board = self.value_normalization_on_board:view(self.batch_size, constants.players_count, 1)
    self.next_round_values_on_board:cmul(normalization_view_on_board:expandAs(self.next_round_values_on_board))
    self.counterfactual_value_memory:add(self.next_round_values_on_board)
  end

  --remembering the values for the next round

  --translating bucket values back to the card values
  self:_bucket_value_to_card_value_aux(
    self.next_round_values:view(self.batch_size * constants.players_count, -1),
    values:view(self.batch_size * constants.players_count, -1))
end

--- Gives the predicted counterfactual values at each evaluated state, given
-- input ranges.
--
-- @{start_computation} must be called first. Each state to be evaluated must
-- be given in the same order that pot sizes were given for that function.
-- Keeps track of iterations internally, so should be called exactly once for
-- every iteration of continual re-solving.
--
-- @param ranges An Nx2xK tensor, where N is the number of states evaluated
-- (must match input to @{start_computation}), 2 is the number of players, and
-- K is the number of private hands. Contains N sets of 2 range vectors.
-- @param values an Nx2xK tensor in which to store the N sets of 2 value vectors
-- which are output
function NextRoundValuePre:get_value(ranges, values)
  assert(ranges and values)
  assert(ranges:size(1) == self.batch_size)
  self.iter = self.iter + 1
  print(self.iter)
  if self.iter == 1 then
    self.next_round_inputs = arguments.Tensor(self.batch_size, self.board_count, (self.bucket_count * constants.players_count + 1)):zero()
    self.next_round_values = arguments.Tensor(self.batch_size, self.board_count, constants.players_count,  self.bucket_count ):zero()
    self.transposed_next_round_values = arguments.Tensor(self.batch_size, constants.players_count, self.board_count, self.bucket_count)
    self.next_round_extended_range = arguments.Tensor(self.batch_size, constants.players_count, self.board_count, self.bucket_count + 1):zero()
    self.next_round_serialized_range = self.next_round_extended_range:view(-1, self.bucket_count + 1)
    self.values_per_board = arguments.Tensor(self.batch_size * constants.players_count, self.board_count, game_settings.hand_count)
    self.range_normalization = arguments.Tensor()
    self.value_normalization = arguments.Tensor(self.batch_size, constants.players_count, self.board_count)

    local den = 0
    assert(self._street <= 3)

    if game_settings.nl then
      den = arguments.stack
    else
      if self._street == 4 then
        den = 48
      elseif self._street == 3 then
        den = 48
      elseif self._street == 2 then
        den = 24
      elseif self._street == 1 then
        den = 10
      else
        den = -1
      end
    end

    --handling pot feature for the nn
    local nn_bet_input = self.pot_sizes:clone():mul(1/den)
    nn_bet_input = nn_bet_input:view(-1, 1):expand(self.batch_size, self.board_count)
    self.next_round_inputs[{{}, {}, {-1}}]:copy(nn_bet_input)
  end

  --computing bucket range in next street for both players at once
  self:_card_range_to_bucket_range(
    ranges:view(self.batch_size * constants.players_count, -1),
    self.next_round_extended_range:view(self.batch_size * constants.players_count, -1))

  self.range_normalization:sum(self.next_round_serialized_range[{{},{1,self.bucket_count}}], 2)
  local rn_view = self.range_normalization:view(self.batch_size, constants.players_count, self.board_count)
  for player = 1, constants.players_count do
    self.value_normalization[{{}, player, {}}]:copy(rn_view[{{}, 3 - player, {}}])
  end

  --eliminating division by zero
  self.range_normalization[torch.eq(self.range_normalization, 0)] = 1
  self.next_round_serialized_range:cdiv(self.range_normalization:expandAs(self.next_round_serialized_range))
  for player = 1, constants.players_count do
    local player_range_index = {(player -1) * self.bucket_count + 1, player * self.bucket_count}
    self.next_round_inputs[{{}, {}, player_range_index}]:copy(self.next_round_extended_range[{{},player, {}, {1, self.bucket_count}}])
  end

  --using nn to compute values
  local serialized_inputs_view= self.next_round_inputs:view(self.batch_size * self.board_count, -1)
  local serialized_values_view= self.next_round_values:view(self.batch_size * self.board_count, -1)

  --computing value in the next round
  self.nn:get_value(serialized_inputs_view, serialized_values_view)

  --normalizing values back according to the orginal range sum
  local normalization_view = self.value_normalization:view(self.batch_size, constants.players_count, self.board_count, 1):transpose(2,3)
  self.next_round_values:cmul(normalization_view:expandAs(self.next_round_values))

  self.transposed_next_round_values:copy(self.next_round_values:transpose(3,2))
  --remembering the values for the next round

  --translating bucket values back to the card values
  self:_bucket_value_to_card_value(
    self.transposed_next_round_values:view(self.batch_size * constants.players_count, -1),
    values:view(self.batch_size * constants.players_count, -1))

end

--- Gives the average counterfactual values on the given board across previous
-- calls to @{get_value}.
--
-- Used to update opponent counterfactual values during re-solving after board
-- cards are dealt.
-- @param board a non-empty vector of board cards
-- @param values a tensor in which to store the values
function NextRoundValuePre:get_value_on_board(board, values)
  --check if we have evaluated correct number of iterations
  assert(self.iter == arguments.cfr_iters )
  local batch_size = values:size(1)
  assert(batch_size == self.batch_size)

  self.range_normalization_memory[torch.eq(self.range_normalization_memory, 0)] = 1
  local serialized_memory_view = self.counterfactual_value_memory:view(-1, self.bucket_count)
  serialized_memory_view:cdiv(self.range_normalization_memory:expandAs(serialized_memory_view))

  self:_bucket_value_to_card_value_on_board(
    board,
    self.counterfactual_value_memory:view(self.batch_size * constants.players_count, -1),
    values:view(self.batch_size * constants.players_count, -1))
end
