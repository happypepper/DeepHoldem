local arguments = require 'Settings.arguments'
local constants = require 'Settings.constants'
local game_settings = require 'Settings.game_settings'
local card_to_string = require 'Game.card_to_string_conversion'
local card_tools = require 'Game.card_tools'
local bucketer = require 'Nn.bucketer'

require 'TerminalEquity.terminal_equity'
require 'Lookahead.lookahead'
require 'Lookahead.resolving'
require 'Nn.value_nn'
require 'Nn.next_round_value'

local current_node = {}

current_node.board = card_to_string:string_to_board('7d7c8s5sQd')
current_node.street = 4
current_node.current_player = constants.players.P2
current_node.bets = arguments.Tensor{8000, 8000}
current_node.num_bets = 0

local te = TerminalEquity()
te:set_board(current_node.board)

local resolving = Resolving(te)

local player_range = card_tools:get_file_range('Lookahead/Tests/ranges/situation-p2.txt')
local opponent_range = card_tools:get_file_range('Lookahead/Tests/ranges/situation-p1.txt')


local player_range_tensor = arguments.Tensor(1,player_range:size(1))
local opponent_range_tensor = arguments.Tensor(1,opponent_range:size(1))

player_range_tensor[1]:copy(player_range)
opponent_range_tensor[1]:copy(opponent_range)
--player_range_tensor[1]:copy(card_tools:get_uniform_range(current_node.board))
--opponent_range_tensor[1]:copy(card_tools:get_uniform_range(current_node.board))

local results = resolving:resolve_first_node(current_node, player_range_tensor, opponent_range_tensor)

for card1 = 1,game_settings.card_count do
  for card2 = card1+1,game_settings.card_count do
    local idx = card_tools:get_hole_index({card1,card2})
    if player_range_tensor[1][idx] > 0 then
      print(card_to_string:card_to_string(card1) .. card_to_string:card_to_string(card2)
            .. ": " .. results.root_cfvs_both_players[1][1][idx])
    end
  end
end
io.read()
for card1 = 1,game_settings.card_count do
  for card2 = card1+1,game_settings.card_count do
    local idx = card_tools:get_hole_index({card1,card2})
    if opponent_range_tensor[1][idx] > 0 then
      print(card_to_string:card_to_string(card1) .. card_to_string:card_to_string(card2))
      print(results.strategy[{{},1,idx}])
    end
  end
end

--print(results.strategy)
--print(results.achieved_cfvs)

--[[
local resolving = Resolving(terminal_equity)

print(results.strategy)
print(results.achieved_cfvs)

]]

--resolving:resolve(current_node, player_range, opponent_range)

--[[
local lookahead = Lookahead()

local current_node = {}
current_node.board = card_to_string:string_to_board('Ks')
current_node.street = 2
current_node.current_player = constants.players.P1
current_node.bets = arguments.Tensor{100, 100}


lookahead:build_lookahead(current_node)
]]

--[[
local starting_ranges = arguments.Tensor(constants.players_count, constants.card_count)
starting_ranges[1]:copy(card_tools:get_random_range(current_node.board, 2))
starting_ranges[2]:copy(card_tools:get_random_range(current_node.board, 4))

lookahead:resolve_first_node(starting_ranges)

lookahead:get_strategy()
]]

--[[
local player_range = card_tools:get_random_range(current_node.board, 2)
local opponent_cfvs = card_tools:get_random_range(current_node.board, 4)

lookahead:resolve(player_range, opponent_cfvs)


lookahead:get_results()
]]
