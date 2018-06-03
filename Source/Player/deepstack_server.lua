--- Performs the main loop for DeepStack.
-- @script deepstack

local arguments = require 'Settings.arguments'
local socket = require("socket")
local constants = require "Settings.constants"

require "ACPC.acpc_game"
require "Player.continual_resolving"

--1.0 create the ACPC game and connect to the server
local acpc_game = ACPCGame()

local continual_resolving = ContinualResolving()

local last_state = nil
local last_node = nil
-- load namespace
-- create a TCP socket and bind it to the local host, at any port
local server = assert(socket.bind("*", 0))
local ip, port = server:getsockname()
print(ip .. ": " .. port)

local client = server:accept()
print("accepted client")

while 1 do
  local line, err = client:receive()
  -- if there was no error, send it back to the client
  if not err then
    print(line)
  else
    print(err)
  end

  local state
  local node
  --2.1 blocks until it's our situation/turn
  state, node = acpc_game:string_to_statenode(line)

  --did a new hand start?
  if not last_state or last_state.hand_number ~= state.hand_number or node.street < last_node.street then
    continual_resolving:start_new_hand(state)
  end
  --2.2 use continual resolving to find a strategy and make an action in the current node
  local adviced_action = continual_resolving:compute_action(node, state)
  local action_id = adviced_action["action"]
  local betsize = adviced_action["raise_amount"]
  print(action_id)
  print(betsize)
  if betsize ~= nil then
    client:send(tostring(betsize))
  elseif action_id == constants.acpc_actions.fold then
    client:send("f")
  elseif action_id == constants.acpc_actions.ccall then
    client:send("c")
  else
    client:send("WTF")
  end
  last_state = state
  last_node = node
  collectgarbage();collectgarbage()
end
