--- Game constants which define the game played by DeepStack.
-- @module game_settings

require 'torch'

local tools = require 'tools'

--leduc defintion
local M = {}
--- the number of card suits in the deck
M.suit_count = 4
--- the number of card ranks in the deck
M.rank_count = 13
--- the total number of cards in the deck
M.card_count = M.suit_count * M.rank_count
--- the number of public cards dealt in the game (revealed after the first
-- betting round)
M.hand_card_count = 2

M.hand_count = tools:choose(M.card_count, M.hand_card_count)

M.board_card_count = {0, 3, 4, 5}

M.limit_bet_sizes = {2, 2, 4, 4}

M.limit_bet_cap = 4

M.nl = true

return M
