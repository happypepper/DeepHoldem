--- Generates neural net training data by solving random poker situations.
-- @module aux_data_generation
local arguments = require 'Settings.arguments'
local game_settings = require 'Settings.game_settings'
local card_generator = require 'DataGeneration.random_card_generator'
local card_to_string_conversion = require 'Game.card_to_string_conversion'
local constants = require 'Settings.constants'
local bucketer = require 'Nn.bucketer'
local card_tools = require 'Game.card_tools'

require 'Nn.bucket_conversion'
require 'Nn.next_round_value_pre'
require 'Nn.value_nn'
require 'DataGeneration.range_generator'
require 'TerminalEquity.terminal_equity'
require 'Lookahead.lookahead'
require 'Lookahead.resolving'
require 'tools'
local M = {}

--- Generates training files by sampling random poker
-- situations and solving them.
--
-- @param train_data_count the number of training examples to generate
function M:generate_data(train_data_count, filename, street)
  --valid data generation
  local timer = torch.Timer()
  timer:reset()

  print('Generating auxiliary data ...')
  self:generate_data_file(train_data_count, filename, street)
  print('gen time: ' .. timer:time().real)

end

--- Generates data files containing examples of random poker situations with
-- counterfactual values from an associated solution.
--
-- Each poker situation is randomly generated using @{range_generator} and
-- @{random_card_generator}. For description of neural net input and target
-- type, see @{net_builder}.
--
-- @param data_count the number of examples to generate
-- @param file_name the prefix of the files where the data is saved (appended
-- with `.inputs`, `.targets`, and `.mask`).
function M:generate_data_file(data_count, file_name, street)
  local range_generator = RangeGenerator()
  local batch_size = arguments.gen_batch_size
  assert(data_count % batch_size == 0, 'data count has to be divisible by the batch size')
  local batch_count = data_count / batch_size

  local target_size = game_settings.hand_count * constants.players_count
  local targets = arguments.Tensor(batch_size, target_size)
  local input_size = game_settings.hand_count * constants.players_count + 1
  local inputs = arguments.Tensor(batch_size, input_size)
  local mask = arguments.Tensor(batch_size, game_settings.hand_count):zero()

  local board = arguments.Tensor()
  local te = TerminalEquity()
  te:set_board(board)
  range_generator:set_board(te, board)

  local bucket_conversion = BucketConversion()
  bucket_conversion:set_board(board)

  local next_round = NextRoundValuePre(ValueNn(street), nil, board)

  local bucket_count = bucketer:get_bucket_count(street)
  local bucketed_target_size = bucket_count * constants.players_count
  local bucketed_input_size = bucket_count * constants.players_count + 1

  local input_batch = arguments.Tensor(arguments.gen_batch_size, bucketed_input_size)
  local target_batch = arguments.Tensor(arguments.gen_batch_size, bucketed_target_size)

  local raw_indexes = {{1, game_settings.hand_count}, {game_settings.hand_count + 1, game_settings.hand_count * 2}}
  local bucket_indexes = {{1, bucket_count}, {bucket_count + 1, bucket_count * 2}}

  for batch = 1, batch_count do
    local timer = torch.Timer()
    timer:reset()

    --generating ranges
    local ranges = arguments.Tensor(constants.players_count, batch_size, game_settings.hand_count)
    for player = 1, constants.players_count do
      range_generator:generate_range(ranges[player])
    end

    --generating pot sizes between ante and stack - 0.1
    local min_pot = {}
    local max_pot = {}

    if game_settings.nl then
      min_pot = {100,200,400,2000,6000}
      max_pot = {100,400,2000,6000,18000}
    else
      if street == 4 then
        min_pot = {2,12,24}
        max_pot = {12,24,48}
      elseif street == 3 then
        min_pot = {2,8,16}
        max_pot = {8,16,24}
      elseif street == 2 then
        min_pot = {2,4,6}
        max_pot = {4,6,10}
      end
    end

    local pot_range = {}

    for i = 1,#min_pot do
      pot_range[i] = max_pot[i] - min_pot[i]
    end
    local random_pot_cats = torch.rand(arguments.gen_batch_size):mul(#min_pot):add(1):floor()
    local random_pot_sizes = torch.rand(arguments.gen_batch_size,1)
    for i = 1, arguments.gen_batch_size do
      random_pot_sizes[i][1] = random_pot_sizes[i][1] * pot_range[random_pot_cats[i]]
      random_pot_sizes[i][1] = random_pot_sizes[i][1] + min_pot[random_pot_cats[i]]
    end

    --pot features are pot sizes normalized between (ante/stack,1)
    local pot_size_features = game_settings.nl and random_pot_sizes:clone():mul(1/arguments.stack) or
        random_pot_sizes:clone():mul(1/max_pot[3])

    --translating ranges to features
    local pot_feature_index =  -1
    inputs[{{}, pot_feature_index}]:copy(pot_size_features)
    input_batch[{{}, pot_feature_index}]:copy(pot_size_features)

    local player_indexes = {{1, game_settings.hand_count}, {game_settings.hand_count + 1, game_settings.hand_count * 2}}
    for player = 1, constants.players_count do
      local player_index = player_indexes[player]
      inputs[{{}, player_index}]:copy(ranges[player])
    end

    for i = 1, arguments.gen_batch_size do
      local next_street_boxes_inputs = arguments.Tensor(1, constants.players_count, game_settings.hand_count):zero()
      local next_street_boxes_outputs = next_street_boxes_inputs:clone()

      for player = 1, constants.players_count do
        local player_index = player_indexes[player]
        next_street_boxes_inputs[{{},player,{}}]:copy(inputs[{i,player_index}])
      end

      next_round:start_computation(random_pot_sizes[i], 1)
      next_round:get_value(next_street_boxes_inputs, next_street_boxes_outputs)

      for player = 1, constants.players_count do
        local player_index = player_indexes[player]
        targets[{i, player_index}]:copy(next_street_boxes_outputs[{{},player,{}}])
      end
    end
    for player = 1, constants.players_count do
      local player_index = raw_indexes[player]
      local bucket_index = bucket_indexes[player]
      bucket_conversion:card_range_to_bucket_range(inputs[{{},player_index}],input_batch[{{}, bucket_index}])
    end
    for player = 1, constants.players_count do
      local player_index = raw_indexes[player]
      local bucket_index = bucket_indexes[player]
      bucket_conversion:hand_cfvs_to_bucket_cfvs(
        inputs[{{}, player_index}],
        targets[{{}, player_index}],
        input_batch[{{}, bucket_index}],
        target_batch[{{}, bucket_index}])
    end

    local basename = file_name .. '-' .. batch
    local train_folder = "xxx/"
    if game_settings.nl then
      train_folder = "NoLimit/"
    else
      train_folder = "Limit/"
    end
    torch.save(arguments.data_path  .. train_folder ..  basename .. '.inputs', input_batch:float())
    torch.save(arguments.data_path  .. train_folder ..  basename .. '.targets', target_batch:float())
  end
end

return M
