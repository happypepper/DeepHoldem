--- Script that trains the neural network.
--
-- Uses data previously generated with @{data_generation_call}.
-- @script main_train

local nnBuilder = require 'Nn.net_builder'
require 'Training.data_stream'
local train = require 'Training.train'
local arguments = require 'Settings.arguments'

if #arg == 0 then
  print("Please specify the street. 1 = preflop, 4 = river")
  return
end

local street = tonumber(arg[1])
local network = nnBuilder:build_net(street)

if arguments.gpu then
  network = network:cuda()
end

local data_stream = DataStream(street)
train:train(network, data_stream, arguments.epoch_count)
