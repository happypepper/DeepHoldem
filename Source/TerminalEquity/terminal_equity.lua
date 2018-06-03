--- Evaluates player equities at terminal nodes of the game's public tree.
-- @classmod terminal_equity

require 'torch'
local evaluator = require 'Game.Evaluation.evaluator'
local game_settings = require 'Settings.game_settings'
local arguments = require 'Settings.arguments'
local card_tools = require 'Game.card_tools'
local constants = require 'Settings.constants'
local card_to_string = require 'Game.card_to_string_conversion'
local tools = require 'tools'

local TerminalEquity = torch.class('TerminalEquity')

--- Constructor
function TerminalEquity:__init()
  self._block_matrix = arguments.Tensor(game_settings.hand_count, game_settings.hand_count):fill(1);
  if game_settings.hand_card_count == 2 then
    for i = 1, game_settings.card_count do
      for j = i+1, game_settings.card_count do
        local idx1 = card_tools:get_hole_index({i,j})
        for k = 1, game_settings.card_count do
          for l = k+1, game_settings.card_count do
            local idx2 = card_tools:get_hole_index({k,l})
            if i == k or i == l or j == k or j == l then
              self._block_matrix[idx1][idx2] = 0
              self._block_matrix[idx2][idx1] = 0
            end
          end
        end
      end
    end
  end
  self.matrix_mem = arguments.Tensor()

  if self._pf_equity == nil then
    local f = assert(io.open("./TerminalEquity/pf_equity.dat", "rb"))
    local data = f:read("*all")
    self._pf_equity = arguments.Tensor(game_settings.hand_count,game_settings.hand_count):fill(0)

    assert(string.len(data) == 1326*1326*4, 'bad length')

    local byteidx = 1
    for i = 1, 1326 do
      for j = 1, 1326 do
        local num = 0
        for k = 0,3 do
          num = num + data:byte(byteidx) * (2 ^ (k * 8))
          byteidx = byteidx + 1
        end
        num = (num > 2147483647) and (num - 4294967296) or num
        -- negative because of how equity matrix is set up
        self._pf_equity[i][j] = -num / 1712304
      end
    end

    f:close()
  end
  self.batch_size = 10
end

--- Constructs the matrix that turns player ranges into showdown equity.
--
-- Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay` is the equity
-- for the first player when no player folds.
--
-- @param board_cards a non-empty vector of board cards
-- @param call_matrix a tensor where the computed matrix is stored
-- @local
function TerminalEquity:get_last_round_call_matrix(board_cards, call_matrix)
  assert(board_cards:dim() == 0 or board_cards:size(1) == 1 or board_cards:size(1) == 2 or board_cards:size(1) == 5,
    'Only Leduc, extended Leduc, and Texas Holdem are supported ' .. board_cards:size(1))

  local strength = evaluator:batch_eval_fast(board_cards)
  --handling hand stregths (winning probs);
  local strength_view_1 = strength:view(game_settings.hand_count, 1):expandAs(call_matrix)
  local strength_view_2 = strength:view(1, game_settings.hand_count):expandAs(call_matrix)

  call_matrix:copy(torch.gt(strength_view_1, strength_view_2))
  call_matrix:csub(torch.lt(strength_view_1, strength_view_2):typeAs(call_matrix))

  self:_handle_blocking_cards(call_matrix, board_cards);
end

--- Constructs the matrix that turns player ranges into showdown equity.
--
-- Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay` is the equity
-- for the first player when no player folds.
--
-- @param board_cards a non-empty vector of board cards
-- @param call_matrix a tensor where the computed matrix is stored
-- @local
function TerminalEquity:get_inner_call_matrix(board_cards, call_matrix)
  assert(board_cards:dim() == 0 or board_cards:size(2) == 1 or board_cards:size(2) == 2 or board_cards:size(2) == 5,
    'Only Leduc, extended Leduc, and Texas Holdem are supported ' .. board_cards:size(2))
  local strength = evaluator:batch_eval_fast(board_cards)
  local num_boards = board_cards:size(1)
  --handling hand stregths (winning probs);
  local strength_view_1 = strength:view(num_boards, game_settings.hand_count, 1):expand(num_boards, game_settings.hand_count, game_settings.hand_count)
  local strength_view_2 = strength:view(num_boards, 1, game_settings.hand_count):expandAs(strength_view_1)

  local possible_mask = torch.lt(strength,0):typeAs(call_matrix)
  for i = 1, num_boards, self.batch_size do
    local indices = {i, i + self.batch_size - 1}
    local sz = self.batch_size
    if i + self.batch_size - 1 > num_boards then
      indices = {i, num_boards}
      sz = num_boards - i + 1
    end
    self.matrix_mem[{{1, sz}}]:copy(torch.gt(strength_view_1[{indices}], strength_view_2[{indices}]))
    self.matrix_mem[{{1, sz}}]:cmul(possible_mask[{indices}]:view(sz, 1, game_settings.hand_count):expand(sz, game_settings.hand_count, game_settings.hand_count))
    self.matrix_mem[{{1, sz}}]:cmul(possible_mask[{indices}]:view(sz, game_settings.hand_count, 1):expand(sz, game_settings.hand_count, game_settings.hand_count))
    call_matrix:add(torch.sum(self.matrix_mem[{{1,sz}}],1))

    self.matrix_mem[{{1, sz}}]:copy(torch.lt(strength_view_1[{indices}], strength_view_2[{indices}]))
    self.matrix_mem[{{1, sz}}]:cmul(possible_mask[{indices}]:view(sz, 1, game_settings.hand_count):expand(sz, game_settings.hand_count, game_settings.hand_count))
    self.matrix_mem[{{1, sz}}]:cmul(possible_mask[{indices}]:view(sz, game_settings.hand_count, 1):expand(sz, game_settings.hand_count, game_settings.hand_count))
    call_matrix:csub(torch.sum(self.matrix_mem[{{1,sz}}],1))
  end

  self:_handle_blocking_cards(call_matrix, board_cards);
end

--- Zeroes entries in an equity matrix that correspond to invalid hands.
--
-- A hand is invalid if it shares any cards with the board.
--
-- @param equity_matrix the matrix to modify
-- @param board a possibly empty vector of board cards
-- @local
function TerminalEquity:_handle_blocking_cards(equity_matrix, board)
  local possible_hand_indexes = card_tools:get_possible_hand_indexes(board);
  local possible_hand_matrix = possible_hand_indexes:view(1, game_settings.hand_count):expandAs(equity_matrix);

  equity_matrix:cmul(possible_hand_matrix);
  possible_hand_matrix = possible_hand_indexes:view(game_settings.hand_count,1):expandAs(equity_matrix);
  equity_matrix:cmul(possible_hand_matrix);

  if game_settings.hand_card_count == 2 then
    equity_matrix:cmul(self._block_matrix)
  elseif game_settings.hand_card_count == 1 then
    for i = 1, game_settings.card_count do
      equity_matrix[i][i] = 0
    end
  end
end

--- Sets the evaluator's fold matrix, which gives the equity for terminal
-- nodes where one player has folded.
--
-- Creates the matrix `B` such that for player ranges `x` and `y`, `x'By` is the equity
-- for the player who doesn't fold
-- @param board a possibly empty vector of board cards
-- @local

function TerminalEquity:_set_fold_matrix(board)
  self.fold_matrix = arguments.Tensor(game_settings.hand_count, game_settings.hand_count);
  self.fold_matrix:fill(1);
  --setting cards that block each other to zero
  self:_handle_blocking_cards(self.fold_matrix, board);
end

--- Sets the evaluator's call matrix, which gives the equity for terminal
-- nodes where no player has folded.
--
-- For nodes in the last betting round, creates the matrix `A` such that for player ranges
-- `x` and `y`, `x'Ay` is the equity for the first player when no player folds. For nodes
-- in the first betting round, gives the weighted average of all such possible matrices.
--
-- @param board a possibly empty vector of board cards
-- @local
-- TODO finish this
function TerminalEquity:_set_call_matrix(board)
  local street = card_tools:board_to_street(board);

  self.equity_matrix = arguments.Tensor(game_settings.hand_count, game_settings.hand_count):zero();
  if street == constants.streets_count then
    --for last round we just return the matrix
    self:get_last_round_call_matrix(board, self.equity_matrix);
  elseif street == 3 or street == 2 then
    --iterate through all possible next round streets
    --TODO(go to the last street)
    local next_round_boards = card_tools:get_last_round_boards(board);
  --  assert(false, 'hey')
    local boards_count = next_round_boards:size(1);

    if self.matrix_mem:dim() ~= 3 or self.matrix_mem:size(2) ~= game_settings.hand_count or self.matrix_mem:size(3) ~= game_settings.hand_count then
      self.matrix_mem = arguments.Tensor(self.batch_size, game_settings.hand_count, game_settings.hand_count)
    end
    self:get_inner_call_matrix(next_round_boards, self.equity_matrix)

    --averaging the values in the call matrix
    local cards_to_come = game_settings.board_card_count[constants.streets_count] - game_settings.board_card_count[street]
    local cards_left = game_settings.card_count - game_settings.hand_card_count*2 - game_settings.board_card_count[street]
    local den = tools:choose(cards_left, cards_to_come)

    local weight_constant = 1/den

    self.equity_matrix:mul(weight_constant);
  elseif street == 1 then
    self.equity_matrix:copy(self._pf_equity)
  else
    --impossible street
    assert(false, 'impossible street ' .. street);
  end
end

function TerminalEquity:get_hand_strengths()
  local a = arguments.Tensor(1, game_settings.hand_count):fill(1)
  return torch.mm(a,self.equity_matrix)
end

--- Sets the board cards for the evaluator and creates its internal data structures.
-- @param board a possibly empty vector of board cards
function TerminalEquity:set_board(board)
  self.board = board
  self:_set_call_matrix(board);
  self:_set_fold_matrix(board);
end

--- Computes (a batch of) counterfactual values that a player achieves at a terminal node
-- where no player has folded.
--
-- @{set_board} must be called before this function.
--
-- @param ranges a batch of opponent ranges in an NxK tensor, where N is the batch size
-- and K is the range size
-- @param result a NxK tensor in which to save the cfvs
function TerminalEquity:call_value( ranges, result )
  result:mm(ranges, self.equity_matrix);
end

--- Computes (a batch of) counterfactual values that a player achieves at a terminal node
-- where a player has folded.
--
-- @{set_board} must be called before this function.
--
-- @param ranges a batch of opponent ranges in an NxK tensor, where N is the batch size
-- and K is the range size
-- @param result A NxK tensor in which to save the cfvs. Positive cfvs are returned, and
-- must be negated if the player in question folded.
function TerminalEquity:fold_value( ranges, result )
  result:mm(ranges, self.fold_matrix);
end

--- Returns the matrix which gives showdown equity for any ranges.
--
-- @{set_board} must be called before this function.
--
-- @return For nodes in the last betting round, the matrix `A` such that for player ranges
-- `x` and `y`, `x'Ay` is the equity for the first player when no player folds. For nodes
-- in the first betting round, the weighted average of all such possible matrices.
function TerminalEquity:get_call_matrix()
  return self.equity_matrix
end

--- Computes the counterfactual values that both players achieve at a terminal node
-- where no player has folded.
--
-- @{set_board} must be called before this function.
--
-- @param ranges a 2xK tensor containing ranges for each player (where K is the range size)
-- @param result a 2xK tensor in which to store the cfvs for each player
function TerminalEquity:tree_node_call_value( ranges, result )
  assert(ranges:dim() == 2)
  assert(result:dim() == 2)
  self:call_value(ranges[1]:view(1,  -1), result[2]:view(1,  -1))
  self:call_value(ranges[2]:view(1,  -1), result[1]:view(1,  -1))
end

--- Computes the counterfactual values that both players achieve at a terminal node
-- where either player has folded.
--
-- @{set_board} must be called before this function.
--
-- @param ranges a 2xK tensor containing ranges for each player (where K is the range size)
-- @param result a 2xK tensor in which to store the cfvs for each player
-- @param folding_player which player folded
function TerminalEquity:tree_node_fold_value( ranges, result, folding_player )
  assert(ranges:dim() == 2)
  assert(result:dim() == 2)
  self:fold_value(ranges[1]:view(1,  -1), result[2]:view(1,  -1))
  self:fold_value(ranges[2]:view(1,  -1), result[1]:view(1,  -1))

  result[folding_player]:mul(-1)
end

TerminalEquity:__init()
--TerminalEquity:set_board(torch.Tensor({48,52,44, 1}))
