local arguments = require 'Settings.arguments'
local constants = require 'Settings.constants'
local card_to_string = require 'Game.card_to_string_conversion'
require 'Tree.tree_builder'
require 'Tree.tree_visualiser'

local builder = PokerTreeBuilder()

local params = {}

params.root_node = {}
params.root_node.board = card_to_string:string_to_board('')
params.root_node.street = 1
params.root_node.current_player = constants.players.P1
params.root_node.bets = arguments.Tensor{50, 100}
params.root_node.num_bets = 0
params.limit_to_street = true

local tree = builder:build_tree(params)

local visualiser = TreeVisualiser()

visualiser:graphviz(tree, "simple_tree")
