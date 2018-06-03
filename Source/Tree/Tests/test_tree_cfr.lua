local arguments = require 'Settings.arguments'
local constants = require 'Settings.constants'
local game_settings = require 'Settings.game_settings'
local card_tools = require 'Game.card_tools'
local card_to_string = require 'Game.card_to_string_conversion'
require 'Tree.tree_builder'
require 'Tree.tree_visualiser'
require 'Tree.tree_values'
require 'Tree.tree_cfr'

local builder = PokerTreeBuilder()

local params = {}

params.root_node = {}
params.root_node.board = card_to_string:string_to_board('6c6d2c2d3h')--card_to_string:string_to_board('Ks')
params.root_node.street = 4
params.root_node.current_player = constants.players.P1
params.root_node.bets = arguments.Tensor{4, 4}
params.root_node.num_bets = 0

local tree = builder:build_tree(params)

local starting_ranges = arguments.Tensor(constants.players_count, game_settings.hand_count)

starting_ranges[1]:copy(card_tools:get_uniform_range(params.root_node.board))
starting_ranges[2]:copy(card_tools:get_uniform_range(params.root_node.board))
--starting_ranges[1]:copy(card_tools:get_random_range(params.root_node.board, 2))
--starting_ranges[2]:copy(card_tools:get_random_range(params.root_node.board, 4))

local tree_cfr = TreeCFR()
tree_cfr:run_cfr(tree, starting_ranges)

print(tree.cf_values)

local tree_values = TreeValues()
tree_values:compute_values(tree, starting_ranges)

print('Exploitability: ' .. tree.exploitability .. '[chips]' )

local visualiser = TreeVisualiser()
visualiser:graphviz(tree, 'simple_tree')
