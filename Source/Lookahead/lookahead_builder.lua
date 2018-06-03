--- Builds the internal data structures of a @{lookahead|Lookahead} object.
-- @classmod lookahead_builder
local arguments = require 'Settings.arguments'
local constants = require 'Settings.constants'
local game_settings = require 'Settings.game_settings'
local tools = require 'tools'
require 'Tree.tree_builder'
require 'Tree.tree_visualiser'
require 'Nn.next_round_value'
require 'Nn.next_round_value_pre'
require 'Nn.value_nn'

local LookaheadBuilder = torch.class('LookaheadBuilder')

--used to load NNs and next_round_value_pre only once
local neural_net = {}
local aux_net = nil
local next_round_pre = nil

--- Constructor
-- @param lookahead the @{lookahead|Lookahead} to generate data structures for
function LookaheadBuilder:__init(lookahead)
  self.lookahead = lookahead
  self.lookahead.ccall_action_index = 1
  self.lookahead.fold_action_index = 2
end

--- Builds the neural net query boxes which estimate counterfactual values
-- at depth-limited states of the lookahead.
-- @local
function LookaheadBuilder:_construct_transition_boxes()
  if self.lookahead.tree.street == constants.streets_count then
    return
  end

  --load neural nets if not already loaded
  local nn = neural_net[self.lookahead.tree.street] or ValueNn(self.lookahead.tree.street)
  neural_net[self.lookahead.tree.street] = nn
  if self.lookahead.tree.street == 1 and game_settings.nl then
    aux_net = aux_net or ValueNn(self.lookahead.tree.street, true)
  end

  self.lookahead.next_street_boxes = nil
  self.lookahead.indices = {}
  self.lookahead.num_pot_sizes = 0

  if self.lookahead.tree.street == 1 then
    self.lookahead.next_street_boxes = next_round_pre or NextRoundValuePre(nn, aux_net, self.lookahead.terminal_equity.board)
    next_round_pre = self.lookahead.next_street_boxes
  else
    self.lookahead.next_street_boxes = NextRoundValue(nn, self.lookahead.terminal_equity.board)
  end

  --create the optimized data structures for batching next_round_value

  for d = 2,self.lookahead.depth do
    if d == 2 and self.lookahead.first_call_transition then
      local before = self.lookahead.num_pot_sizes
      self.lookahead.num_pot_sizes = self.lookahead.num_pot_sizes + 1
      self.lookahead.indices[d] = {before + 1, self.lookahead.num_pot_sizes}
    elseif not game_settings.nl and (d > 2 or self.lookahead.first_call_transition) then
      local before = self.lookahead.num_pot_sizes
      self.lookahead.num_pot_sizes = self.lookahead.num_pot_sizes + (self.lookahead.pot_size[d][2]:size(1)) * self.lookahead.pot_size[d][2]:size(2)
      self.lookahead.indices[d] = {before + 1, self.lookahead.num_pot_sizes}
    elseif self.lookahead.pot_size[d][2]:size(1) > 1 then
      local before = self.lookahead.num_pot_sizes
      self.lookahead.num_pot_sizes = self.lookahead.num_pot_sizes + (self.lookahead.pot_size[d][2]:size(1) - 1) * self.lookahead.pot_size[d][2]:size(2)
      self.lookahead.indices[d] = {before + 1, self.lookahead.num_pot_sizes}
    end
  end

  if self.lookahead.num_pot_sizes == 0 then
    return
  end

  self.lookahead.next_round_pot_sizes = arguments.Tensor(self.lookahead.num_pot_sizes):zero()

  self.lookahead.action_to_index = {}
  for d = 2,self.lookahead.depth do
    local parent_indices = {1, -2}
    if self.lookahead.indices[d] ~= nil then
      if d == 2 then
        parent_indices = {1, 1}
      elseif not game_settings.nl then
        parent_indices = {}
      end
      self.lookahead.next_round_pot_sizes[{self.lookahead.indices[d]}]:copy(self.lookahead.pot_size[d][{2, parent_indices, {}, 1, 1, 1}])
      if d <= 3 then
        if d == 2 then
          assert(self.lookahead.indices[d][1] == self.lookahead.indices[d][2])
          self.lookahead.action_to_index[constants.actions.ccall] = self.lookahead.indices[d][1]
        else
          assert(self.lookahead.pot_size[d][{2, parent_indices}]:size(2) == 1, 'bad num_indices: ')
          for parent_action_idx = 1, self.lookahead.pot_size[d][2]:size(1) do
            local action_id = self.lookahead.parent_action_id[parent_action_idx]
            assert(self.lookahead.action_to_index[action_id] == nil)
            self.lookahead.action_to_index[action_id] = self.lookahead.indices[d][1] + parent_action_idx - 1
          end
        end
      end
    end
  end
  if self.lookahead.action_to_index[constants.actions.ccall] == nil then
    print(self.lookahead.action_to_index)
    print(self.lookahead.parent_action_id)
    assert(false)
  end
  self.lookahead.next_street_boxes:start_computation(self.lookahead.next_round_pot_sizes, self.lookahead.batch_size)
  self.lookahead.next_street_boxes_inputs = arguments.Tensor(self.lookahead.num_pot_sizes, self.lookahead.batch_size, constants.players_count, game_settings.hand_count):zero()
  self.lookahead.next_street_boxes_outputs = self.lookahead.next_street_boxes_inputs:clone()
end

--- Computes the number of nodes at each depth of the tree.
--
-- Used to find the size for the tensors which store lookahead data.
-- @local
function LookaheadBuilder:_compute_structure()

  assert(self.lookahead.tree.street >= 1 and self.lookahead.tree.street <= constants.streets_count)

  self.lookahead.regret_epsilon = 1.0 / 1000000000

  --which player acts at particular depth
  self.lookahead.acting_player = torch.Tensor(self.lookahead.depth+1):fill(-1)
  self.lookahead.acting_player[1] = 1 --in lookahead, 1 does not stand for player IDs, it's just the first player to act
  for d=2,self.lookahead.depth+1 do
    self.lookahead.acting_player[d] = 3 - self.lookahead.acting_player[d-1]
  end


  self.lookahead.bets_count[-1] = 1
  self.lookahead.bets_count[0] = 1
  self.lookahead.nonallinbets_count[-1] = 1
  self.lookahead.nonallinbets_count[0] = 1
  self.lookahead.terminal_actions_count[-1] = 0
  self.lookahead.terminal_actions_count[0] = 0
  self.lookahead.actions_count[-1] = 1
  self.lookahead.actions_count[0] = 1

  --compute the node counts
  self.lookahead.nonterminal_nodes_count = {}
  self.lookahead.nonterminal_nonallin_nodes_count = {}
  self.lookahead.all_nodes_count = {}
  self.lookahead.allin_nodes_count = {}
  self.lookahead.inner_nodes_count = {}

  self.lookahead.nonterminal_nodes_count[1] = 1
  self.lookahead.nonterminal_nodes_count[2] = self.lookahead.bets_count[1]
  self.lookahead.nonterminal_nonallin_nodes_count[0] = 1
  self.lookahead.nonterminal_nonallin_nodes_count[1] = 1
  self.lookahead.nonterminal_nonallin_nodes_count[2] = self.lookahead.nonterminal_nodes_count[2]
  if game_settings.nl then
    self.lookahead.nonterminal_nonallin_nodes_count[2] = self.lookahead.nonterminal_nonallin_nodes_count[2] - 1
  end
  self.lookahead.all_nodes_count[1] = 1
  self.lookahead.all_nodes_count[2] = self.lookahead.actions_count[1]
  self.lookahead.allin_nodes_count[1] = 0
  self.lookahead.allin_nodes_count[2] = 1
  self.lookahead.inner_nodes_count[1] = 1
  self.lookahead.inner_nodes_count[2] = 1

  for d=2,self.lookahead.depth do
    self.lookahead.all_nodes_count[d+1] = self.lookahead.nonterminal_nonallin_nodes_count[d-1] * self.lookahead.bets_count[d-1] * self.lookahead.actions_count[d]
    self.lookahead.allin_nodes_count[d+1] = self.lookahead.nonterminal_nonallin_nodes_count[d-1] * self.lookahead.bets_count[d-1] * 1

    self.lookahead.nonterminal_nodes_count[d+1] = self.lookahead.nonterminal_nonallin_nodes_count[d-1] * self.lookahead.nonallinbets_count[d-1] * self.lookahead.bets_count[d]
    self.lookahead.nonterminal_nonallin_nodes_count[d+1] = self.lookahead.nonterminal_nonallin_nodes_count[d-1] * self.lookahead.nonallinbets_count[d-1] * self.lookahead.nonallinbets_count[d]
  end
end

--- Builds the tensors that store lookahead data during re-solving.
function LookaheadBuilder:construct_data_structures()

  self:_compute_structure()

  --lookahead main data structures
  --all the structures are per-layer tensors, that is, each layer holds the data in n-dimensional tensors
  self.lookahead.pot_size = {}
  self.lookahead.ranges_data = {}
  self.lookahead.average_strategies_data = {}
  self.lookahead.current_strategy_data = {}
  self.lookahead.cfvs_data = {}
  self.lookahead.average_cfvs_data = {}
  self.lookahead.regrets_data = {}
  self.lookahead.current_regrets_data = {}
  self.lookahead.positive_regrets_data = {}
  self.lookahead.placeholder_data = {}
  self.lookahead.regrets_sum = {}
  self.lookahead.empty_action_mask = {} --used to mask empty actions
  --used to hold and swap inner (nonterminal) nodes when doing some transpose operations
  self.lookahead.inner_nodes = {}
  self.lookahead.inner_nodes_p1 = {}
  self.lookahead.swap_data = {}


  --create the data structure for the first two layers

  --data structures [actions x parent_action x grandparent_id x batch x players x range]
  self.lookahead.ranges_data[1] = arguments.Tensor(1, 1, 1, self.lookahead.batch_size, constants.players_count, game_settings.hand_count):fill(1.0 / game_settings.hand_count)
  self.lookahead.ranges_data[2] = arguments.Tensor(self.lookahead.actions_count[1], 1, 1, self.lookahead.batch_size, constants.players_count, game_settings.hand_count):fill(1.0 / game_settings.hand_count)
  self.lookahead.pot_size[1] = self.lookahead.ranges_data[1]:clone():fill(0)
  self.lookahead.pot_size[2] = self.lookahead.ranges_data[2]:clone():fill(0)
  self.lookahead.cfvs_data[1] = self.lookahead.ranges_data[1]:clone():fill(0)
  self.lookahead.cfvs_data[2] = self.lookahead.ranges_data[2]:clone():fill(0)
  self.lookahead.average_cfvs_data[1] = self.lookahead.ranges_data[1]:clone():fill(0)
  self.lookahead.average_cfvs_data[2] = self.lookahead.ranges_data[2]:clone():fill(0)
  self.lookahead.placeholder_data[1] = self.lookahead.ranges_data[1]:clone():fill(0)
  self.lookahead.placeholder_data[2] = self.lookahead.ranges_data[2]:clone():fill(0)

  --data structures for one player [actions x parent_action x grandparent_id x batch x 1 x range]
  self.lookahead.average_strategies_data[1] = nil
  self.lookahead.average_strategies_data[2] = arguments.Tensor(self.lookahead.actions_count[1], 1, 1, self.lookahead.batch_size, game_settings.hand_count):fill(0)
  self.lookahead.current_strategy_data[1] = nil
  self.lookahead.current_strategy_data[2] = self.lookahead.average_strategies_data[2]:clone():fill(0)
  self.lookahead.regrets_data[1] = nil
  self.lookahead.regrets_data[2] = self.lookahead.average_strategies_data[2]:clone():fill(0)
  self.lookahead.current_regrets_data[1] = nil
  self.lookahead.current_regrets_data[2] = self.lookahead.average_strategies_data[2]:clone():fill(0)
  self.lookahead.positive_regrets_data[1] = nil
  self.lookahead.positive_regrets_data[2] = self.lookahead.average_strategies_data[2]:clone():fill(0)
  self.lookahead.empty_action_mask[1] = nil
  self.lookahead.empty_action_mask[2] = self.lookahead.average_strategies_data[2]:clone():fill(1)

  --data structures for summing over the actions [1 x parent_action x grandparent_id x batch x range]
  self.lookahead.regrets_sum[1] = arguments.Tensor(1, 1, 1, self.lookahead.batch_size, game_settings.hand_count):fill(0)
  self.lookahead.regrets_sum[2] = arguments.Tensor(1, self.lookahead.bets_count[1], 1, self.lookahead.batch_size, game_settings.hand_count):fill(0)

  --data structures for inner nodes (not terminal nor allin) [bets_count x parent_nonallinbetscount x gp_id x batch x players x range]
  self.lookahead.inner_nodes[1] = arguments.Tensor(1, 1, 1, self.lookahead.batch_size, constants.players_count, game_settings.hand_count):fill(0)
  self.lookahead.swap_data[1] = self.lookahead.inner_nodes[1]:transpose(2,3):clone()
  self.lookahead.inner_nodes_p1[1] = arguments.Tensor(1, 1, 1, self.lookahead.batch_size, 1, game_settings.hand_count):fill(0)

  if self.lookahead.depth > 2 then
    self.lookahead.inner_nodes[2] = arguments.Tensor(self.lookahead.bets_count[1], 1, 1, self.lookahead.batch_size, constants.players_count, game_settings.hand_count):fill(0)
    self.lookahead.swap_data[2] = self.lookahead.inner_nodes[2]:transpose(2,3):clone()
    self.lookahead.inner_nodes_p1[2] = arguments.Tensor(self.lookahead.bets_count[1], 1, 1, self.lookahead.batch_size, 1, game_settings.hand_count):fill(0)
  end


  --create the data structures for the rest of the layers
  for d=3,self.lookahead.depth do

    --data structures [actions x parent_action x grandparent_id x batch x players x range]
    self.lookahead.ranges_data[d] = arguments.Tensor(self.lookahead.actions_count[d-1], self.lookahead.bets_count[d-2], self.lookahead.nonterminal_nonallin_nodes_count[d-2], self.lookahead.batch_size, constants.players_count, game_settings.hand_count):fill(0)
    self.lookahead.cfvs_data[d] = self.lookahead.ranges_data[d]:clone()
    self.lookahead.placeholder_data[d] = self.lookahead.ranges_data[d]:clone()
    self.lookahead.pot_size[d] = self.lookahead.ranges_data[d]:clone():fill(arguments.stack)

    --data structures [actions x parent_action x grandparent_id x batch x 1 x range]
    self.lookahead.average_strategies_data[d] = arguments.Tensor(self.lookahead.actions_count[d-1], self.lookahead.bets_count[d-2], self.lookahead.nonterminal_nonallin_nodes_count[d-2], self.lookahead.batch_size, game_settings.hand_count):fill(0)
    self.lookahead.current_strategy_data[d] = self.lookahead.average_strategies_data[d]:clone()
    self.lookahead.regrets_data[d] = self.lookahead.average_strategies_data[d]:clone():fill(self.lookahead.regret_epsilon)
    self.lookahead.current_regrets_data[d] = self.lookahead.average_strategies_data[d]:clone():fill(0)
    self.lookahead.empty_action_mask[d] = self.lookahead.average_strategies_data[d]:clone():fill(1)
    self.lookahead.positive_regrets_data[d] = self.lookahead.regrets_data[d]:clone()

    --data structures [1 x parent_action x grandparent_id x batch x players x range]
    self.lookahead.regrets_sum[d] = arguments.Tensor(1, self.lookahead.bets_count[d-2], self.lookahead.nonterminal_nonallin_nodes_count[d-2], self.lookahead.batch_size, constants.players_count, game_settings.hand_count):fill(0)

    --data structures for the layers except the last one
    if d < self.lookahead.depth then
      self.lookahead.inner_nodes[d] = arguments.Tensor(self.lookahead.bets_count[d-1], self.lookahead.nonallinbets_count[d-2], self.lookahead.nonterminal_nonallin_nodes_count[d-2], self.lookahead.batch_size, constants.players_count, game_settings.hand_count):fill(0)
      self.lookahead.inner_nodes_p1[d] = arguments.Tensor(self.lookahead.bets_count[d-1], self.lookahead.nonallinbets_count[d-2], self.lookahead.nonterminal_nonallin_nodes_count[d-2], self.lookahead.batch_size, 1, game_settings.hand_count):fill(0)

      self.lookahead.swap_data[d] = self.lookahead.inner_nodes[d]:transpose(2, 3):clone()
    end
  end

  --create the optimized data structures for terminal equity
  self.lookahead.term_call_indices = {}
  self.lookahead.num_term_call_nodes = 0
  self.lookahead.term_fold_indices = {}
  self.lookahead.num_term_fold_nodes = 0

  -- calculate term_call_indices
  for d = 2,self.lookahead.depth do
    if self.lookahead.tree.street ~= constants.streets_count then
      if game_settings.nl and (d>2 or self.lookahead.first_call_terminal) then
        local before = self.lookahead.num_term_call_nodes
        self.lookahead.num_term_call_nodes = self.lookahead.num_term_call_nodes + self.lookahead.ranges_data[d][2][-1]:size(1)
        self.lookahead.term_call_indices[d] = {before + 1, self.lookahead.num_term_call_nodes}
      end
    else
      if d>2 or self.lookahead.first_call_terminal then
        local before = self.lookahead.num_term_call_nodes
        self.lookahead.num_term_call_nodes = self.lookahead.num_term_call_nodes + self.lookahead.ranges_data[d][2]:size(1) * self.lookahead.ranges_data[d][2]:size(2)
        self.lookahead.term_call_indices[d] = {before + 1, self.lookahead.num_term_call_nodes}
      end
    end
  end

  -- calculate term_fold_indices
  for d = 2,self.lookahead.depth do
    local before = self.lookahead.num_term_fold_nodes
    self.lookahead.num_term_fold_nodes = self.lookahead.num_term_fold_nodes + self.lookahead.ranges_data[d][1]:size(1) * self.lookahead.ranges_data[d][1]:size(2)
    self.lookahead.term_fold_indices[d] = {before + 1, self.lookahead.num_term_fold_nodes}
  end

  self.lookahead.ranges_data_call = arguments.Tensor(self.lookahead.num_term_call_nodes, self.lookahead.batch_size, constants.players_count, game_settings.hand_count)
  self.lookahead.ranges_data_fold = arguments.Tensor(self.lookahead.num_term_fold_nodes, self.lookahead.batch_size, constants.players_count, game_settings.hand_count)

  self.lookahead.cfvs_data_call = arguments.Tensor(self.lookahead.num_term_call_nodes, self.lookahead.batch_size, constants.players_count, game_settings.hand_count)
  self.lookahead.cfvs_data_fold = arguments.Tensor(self.lookahead.num_term_fold_nodes, self.lookahead.batch_size, constants.players_count, game_settings.hand_count)
end

function LookaheadBuilder:reset()
  for d = 1, self.lookahead.depth do
    if self.lookahead.ranges_data[d] ~= nil then
      self.lookahead.ranges_data[d]:fill(1.0 / game_settings.hand_count)
    end
    if self.lookahead.average_strategies_data[d] ~= nil then
      self.lookahead.average_strategies_data[d]:fill(0)
    end
    if self.lookahead.current_strategy_data[d] ~= nil then
      self.lookahead.current_strategy_data[d]:fill(0)
    end
    if self.lookahead.cfvs_data[d] ~= nil then
      self.lookahead.cfvs_data[d]:fill(0)
    end
    if self.lookahead.average_cfvs_data[d] ~= nil then
      self.lookahead.average_cfvs_data[d]:fill(0)
    end
    if self.lookahead.regrets_data[d] ~= nil then
      self.lookahead.regrets_data[d]:fill(0)
    end
    if self.lookahead.current_regrets_data[d] ~= nil then
      self.lookahead.current_regrets_data[d]:fill(0)
    end
    if self.lookahead.positive_regrets_data[d] ~= nil then
      self.lookahead.positive_regrets_data[d]:fill(0)
    end
    if self.lookahead.placeholder_data[d] ~= nil then
      self.lookahead.placeholder_data[d]:fill(0)
    end
    if self.lookahead.regrets_sum[d] ~= nil then
      self.lookahead.regrets_sum[d]:fill(0)
    end
    if self.lookahead.inner_nodes[d] ~= nil then
      self.lookahead.inner_nodes[d]:fill(0)
    end
    if self.lookahead.inner_nodes_p1[d] ~= nil then
      self.lookahead.inner_nodes_p1[d]:fill(0)
    end
    if self.lookahead.swap_data[d] ~= nil then
      self.lookahead.swap_data[d]:fill(0)
    end
  end
  if self.lookahead.next_street_boxes ~= nil then
    self.lookahead.next_street_boxes.iter = 0
    self.lookahead.next_street_boxes:start_computation(self.lookahead.next_round_pot_sizes, self.lookahead.batch_size)
  end
end

--- Traverses the tree to fill in lookahead data structures that summarize data
-- contained in the tree.
--
-- For example, saves pot sizes and numbers of actions at each lookahead state.
--
-- @param node the current node of the public tree
-- @param layer the depth of the current node
-- @param action_id the index of the action that led to this node
-- @param parent_id the index of the current node's parent
-- @param gp_id the index of the current node's grandparent
-- @local
function LookaheadBuilder:set_datastructures_from_tree_dfs(node, layer, action_id, parent_id, gp_id, cur_action_id, parent_action_id)

  --fill the potsize
  assert(node.pot)
  self.lookahead.pot_size[layer][{action_id, parent_id, gp_id, {}, {}}] = node.pot
  if layer == 3 and cur_action_id == constants.actions.ccall then
    self.lookahead.parent_action_id[parent_id] = parent_action_id
  end

  node.lookahead_coordinates = arguments.Tensor({action_id, parent_id, gp_id})

  --transition call cannot be allin call
  if node.current_player == constants.players.chance then
    assert(parent_id <= self.lookahead.nonallinbets_count[layer-2])
  end

  if layer < self.lookahead.depth + 1 then
    local gp_nonallinbets_count = self.lookahead.nonallinbets_count[layer-2]
    local prev_layer_terminal_actions_count = self.lookahead.terminal_actions_count[layer-1]
    local gp_terminal_actions_count = self.lookahead.terminal_actions_count[layer-2]
    local prev_layer_bets_count = 0

    prev_layer_bets_count = self.lookahead.bets_count[layer - 1]

    --compute next coordinates for parent and grandparent
    local next_parent_id = action_id - prev_layer_terminal_actions_count
    local next_gp_id = (gp_id - 1) * gp_nonallinbets_count + (parent_id)

    if (not node.terminal) and (node.current_player ~= constants.players.chance) then

      --parent is not an allin raise
      assert(parent_id <= self.lookahead.nonallinbets_count[layer-2])

      --do we need to mask some actions for that node? (that is, does the node have fewer children than the max number of children for any node on this layer)
      local node_with_empty_actions = (#node.children < self.lookahead.actions_count[layer])

      if node_with_empty_actions then
        --we need to mask nonexisting padded bets
        assert(layer > 1)

        local terminal_actions_count = self.lookahead.terminal_actions_count[layer]
        assert(terminal_actions_count == 2)

        local existing_bets_count = #node.children - terminal_actions_count

        --allin situations
        if existing_bets_count == 0 then
          assert(action_id == self.lookahead.actions_count[layer-1])
        end

        for child_id = 1,terminal_actions_count do
          local child_node = node.children[child_id]
          --go deeper
          self:set_datastructures_from_tree_dfs(child_node, layer+1, child_id, next_parent_id, next_gp_id, node.actions[child_id], cur_action_id)
        end

        --we need to make sure that even though there are fewer actions, the last action/allin is has the same last index as if we had full number of actions
        --we manually set the action_id as the last action (allin)
        for b = 1, existing_bets_count do
          self:set_datastructures_from_tree_dfs(node.children[#node.children-b+1], layer+1, self.lookahead.actions_count[layer]-b+1, next_parent_id, next_gp_id, node.actions[#node.children-b+1], cur_action_id)
        end

        --mask out empty actions
        self.lookahead.empty_action_mask[layer+1][{{terminal_actions_count+1,-(existing_bets_count+1)}, next_parent_id, next_gp_id, {}}] = 0

      else
        --node has full action count, easy to handle
        for child_id = 1,#node.children do
          local child_node = node.children[child_id]
          --go deeper
          self:set_datastructures_from_tree_dfs(child_node, layer+1, child_id, next_parent_id, next_gp_id, node.actions[child_id], cur_action_id)
        end
      end
    end
  end
end


--- Builds the lookahead's internal data structures using the public tree.
-- @param tree the public tree used to construct the lookahead
function LookaheadBuilder:build_from_tree(tree)

  self.lookahead.tree = tree
  self.lookahead.depth = tree.depth

  --per layer information about tree actions
  --per layer actions are the max number of actions for any of the nodes on the layer
  self.lookahead.bets_count = {}
  self.lookahead.nonallinbets_count = {}
  self.lookahead.terminal_actions_count = {}
  self.lookahead.actions_count = {}

  self.lookahead.first_call_terminal = self.lookahead.tree.children[2].terminal
  self.lookahead.first_call_transition = self.lookahead.tree.children[2].current_player == constants.players.chance
  self.lookahead.first_call_check = (not self.lookahead.first_call_terminal) and (not self.lookahead.first_call_transition)

  self:_compute_tree_structures({tree}, 1)
  --construct the initial data structures using the bet counts
  self:construct_data_structures()

  -- action ids for first
  self.lookahead.parent_action_id = {}

  --traverse the tree and fill the datastructures (pot sizes, non-existin actions, ...)
  --node, layer, action, parent_action, gp_id
  self:set_datastructures_from_tree_dfs(tree, 1, 1, 1, 1, -100)

  --set additional info
  assert(self.lookahead.terminal_actions_count[1] == 1 or self.lookahead.terminal_actions_count[1] == 2)

  --we mask out fold as a possible action when check is for free, due to
  --1) fewer actions means faster convergence
  --2) we need to make sure prob of free fold is zero because ACPC dealer changes such action to check
  if self.lookahead.tree.bets[1] == self.lookahead.tree.bets[2] then
    --TODO fix this
    self.lookahead.empty_action_mask[2][1]:fill(0)
  end

  --construct the neural net query boxes
  self:_construct_transition_boxes()

end

--- Computes the maximum number of actions at each depth of the tree.
--
-- Used to find the size for the tensors which store lookahead data. The
-- maximum number of actions is used so that every node at that depth can
-- fit in the same tensor.
-- @param current_layer a list of tree nodes at the current depth
-- @param current_depth the depth of the current tree nodes
-- @local
function LookaheadBuilder:_compute_tree_structures(current_layer, current_depth)

  local layer_actions_count = 0
  local layer_terminal_actions_count = 0
  local next_layer = {}

  for n = 1,#current_layer do
    local node = current_layer[n]
    layer_actions_count = math.max(layer_actions_count, #node.children)

    local node_terminal_actions_count = 0
    for c = 1,#current_layer[n].children do
      if node.children[c].terminal or node.children[c].current_player == constants.players.chance then
        node_terminal_actions_count = node_terminal_actions_count + 1
      end
    end

    layer_terminal_actions_count = math.max(layer_terminal_actions_count, node_terminal_actions_count)

    --add children of the node to the next layer for later pass of BFS
    if not node.terminal  then
      for c = 1,#node.children do
        table.insert(next_layer, node.children[c])
      end
    end
  end

  assert((layer_actions_count == 0) == (#next_layer == 0))
  assert((layer_actions_count == 0) == (current_depth == self.lookahead.depth))

  --set action and bet counts
  self.lookahead.bets_count[current_depth] = layer_actions_count - layer_terminal_actions_count


  self.lookahead.nonallinbets_count[current_depth] = layer_actions_count - layer_terminal_actions_count
  if game_settings.nl then
    --remove allin
    self.lookahead.nonallinbets_count[current_depth] = self.lookahead.nonallinbets_count[current_depth] - 1
  end
  --if no alllin...
  if layer_actions_count == 2 then
    assert(layer_actions_count == layer_terminal_actions_count, layer_terminal_actions_count .. " " .. current_depth)
    self.lookahead.nonallinbets_count[current_depth] = 0
  end

  self.lookahead.terminal_actions_count[current_depth] = layer_terminal_actions_count
  self.lookahead.actions_count[current_depth] = layer_actions_count

  if #next_layer > 0 then
    assert(layer_actions_count >= 2)
    --go deeper
    self:_compute_tree_structures(next_layer, current_depth + 1)
  end

end
