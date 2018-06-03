local arguments = require 'Settings.arguments'
local constants = require "Settings.constants"
require "ACPC.acpc_game"

--1.0 create the ACPC game and connect to the server

local port = 0
if #arg > 0 then
  port = tonumber(arg[1])
else
  print("need port")
  return
end

local acpc_game = ACPCGame()
acpc_game:connect(arguments.acpc_server, port)

--2.0 main loop that waits for a situation where we act and then chooses an action
while true do
  local state
  local node

  --2.1 blocks until it's our situation/turn
  state, node = acpc_game:get_next_situation()

  --print(state)
  --io.read()
  --print(node)
  --io.read()

  print("input action:")
  local action = io.read()

  local acpc_action = nil

  if action == "f" then
    acpc_action = {action = constants.acpc_actions.fold}
  elseif action == "c" then
    acpc_action = {action = constants.acpc_actions.ccall}
  else
    local amount = tonumber(action)
    acpc_action = {action = constants.acpc_actions.raise, raise_amount = amount}
  end

  --2.3 send the action to the dealer
  acpc_game:play_action(acpc_action)

  collectgarbage();collectgarbage()
end
