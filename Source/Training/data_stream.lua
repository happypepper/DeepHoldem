--- Handles the data used for neural net training and validation.
-- @classmod data_stream

require 'torch'
local arguments = require 'Settings.arguments'
local game_settings = require 'Settings.game_settings'
local bucketer = require 'Nn.bucketer'
local constants = require 'Settings.constants'
local DataStream = torch.class('DataStream')

-- Lua implementation of PHP scandir function
function DataStream:_scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -a "'..directory..'"')
    for filename in pfile:lines() do
        i = i + 1
        t[filename] = 1
    end
    pfile:close()
    return t
end

--- Constructor.
--
-- Reads the data from training and validation files generated with
-- @{data_generation_call.generate_data}.
function DataStream:__init(street)
  self.folder = "xxx/"
  if street == 4 then
    self.folder = "river/"
  elseif street == 3 then
    self.folder = "turn/"
  elseif street == 2 then
    self.folder = "flop/"
  elseif street == 1 then
    self.folder = "preflop-aux/"
  end
  self.path = arguments.data_path
  if game_settings.nl then
    self.path = self.path .. "NoLimit/"
  else
    self.path = self.path .. "Limit/"
  end
  self.path = self.path .. self.folder
  filenames = self:_scandir(self.path)
  local numfiles = 0

  local goodfiles = {}

  for filename,_ in pairs(filenames) do
    local res = string.find(filename, ".inputs")
    if res ~= nil then
      local targetname = filename:sub(0,res) .. "targets"
      if filenames[targetname] ~= nil then
        numfiles = numfiles + 1
        goodfiles[numfiles] = filename:sub(0,res)
      end
    end
  end

  self.goodfiles = goodfiles
  print(numfiles .. " good files")

  self.bucket_count = bucketer:get_bucket_count(street)
  self.target_size = self.bucket_count * constants.players_count
  self.input_size = self.bucket_count * constants.players_count + 1

  local num_train = math.floor(numfiles * 0.9)
  local num_valid = numfiles - num_train

  local train_count = num_train * arguments.gen_batch_size
  local valid_count = num_valid * arguments.gen_batch_size

  self.train_data_count = train_count
  assert(self.train_data_count >= arguments.train_batch_size, 'Training data count has to be greater than a train batch size!')
  self.train_batch_count = self.train_data_count / arguments.train_batch_size
  self.valid_data_count = valid_count
  assert(self.valid_data_count >= arguments.train_batch_size, 'Validation data count has to be greater than a train batch size!')
  self.valid_batch_count = self.valid_data_count / arguments.train_batch_size

  --loading train data

  --transfering data to gpu if needed
end

--- Gives the number of batches of validation data.
--
-- Batch size is defined by @{arguments.train_batch_size}.
-- @return the number of batches
function DataStream:get_valid_batch_count()
  return self.valid_batch_count
end

--- Gives the number of batches of training data.
--
-- Batch size is defined by @{arguments.train_batch_size}
-- @return the number of batches
function DataStream:get_train_batch_count()
  return self.train_batch_count
end

function DataStream:shuffle(tbl, n)
  for i = n, 1, -1 do
    local rand = math.random(n)
    tbl[i], tbl[rand] = tbl[rand], tbl[i]
  end
  return tbl
end

--- Randomizes the order of training data.
--
-- Done so that the data is encountered in a different order for each epoch.
function DataStream:start_epoch()
  --data are shuffled each epoch]
  self:shuffle(self.goodfiles, self.train_data_count / arguments.gen_batch_size)
end

--- Returns a batch of data from a specified data set.
-- @param inputs the inputs set for the given data set
-- @param targets the targets set for the given data set
-- @param mask the masks set for the given data set
-- @param batch_index the index of the batch to return
-- @return the inputs set for the batch
-- @return the targets set for the batch
-- @return the masks set for the batch
-- @local
function  DataStream:get_batch(batch_index)

  local inputs = arguments.Tensor(arguments.train_batch_size, self.input_size)
  local targets = arguments.Tensor(arguments.train_batch_size, self.target_size)
  local masks = arguments.Tensor(arguments.train_batch_size, self.target_size):zero()

  for i = 1, arguments.train_batch_size / arguments.gen_batch_size do
    local idx = (batch_index - 1) * arguments.train_batch_size / arguments.gen_batch_size + i
    idx = math.floor(idx + 0.1)
    local filebase = self.goodfiles[idx]

    local inputname = filebase .. "inputs"
    local targetname = filebase .. "targets"

    local input_batch = torch.load(self.path .. inputname)
    local target_batch = torch.load(self.path .. targetname)

    data_index = {{(i - 1) * arguments.gen_batch_size + 1, i * arguments.gen_batch_size}, {}}

    inputs[data_index]:copy(input_batch)
    targets[data_index]:copy(target_batch)
    masks[data_index][torch.gt(input_batch[{{}, {1, self.bucket_count * 2}}], 0)] = 1
  end

  if arguments.gpu then
    inputs = inputs:cuda()
    targets = targets:cuda()
    masks = masks:cuda()
  end
  return inputs, targets, masks
end

--- Returns a batch of data from the training set.
-- @param batch_index the index of the batch to return
-- @return the inputs set for the batch
-- @return the targets set for the batch
-- @return the masks set for the batch
function  DataStream:get_train_batch(batch_index)
    return self:get_batch(batch_index)
end

--- Returns a batch of data from the validation set.
-- @param batch_index the index of the batch to return
-- @return the inputs set for the batch
-- @return the targets set for the batch
-- @return the masks set for the batch
function  DataStream:get_valid_batch(batch_index)
  return self:get_batch(self.train_batch_count + batch_index)
end
