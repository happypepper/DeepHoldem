def acpcify_board(board):
  if len(board) == 6:
    return board
  if len(board) == 8:
    return board[:6] + "/" + board[6:]
  if len(board) == 10:
    return board[:6] + "/" + board[6:8] + "/" + board[8:]
  return "WTF"

def acpcify_actions(actions):
    actions = actions.replace("b","r")
    actions = actions.replace("k","c")
    streets = actions.split("/")
    max_bet = 0
    for i, street_actions in enumerate(streets):
      bets = street_actions.split("r")
      max_street_bet = max_bet
      for j, betstr in enumerate(bets):
        try:
          flag = False
          if len(betstr) > 1 and betstr[-1] == 'c':
            flag = True
            betstr = betstr.replace("c","")
          bet = int(betstr)
          bet += max_bet
          max_street_bet = max(max_street_bet, bet)
          bets[j] = str(bet)
          if flag:
            bets[j] += "c"
          bets[j] = "r" + bets[j]
        except ValueError:
          continue
      max_bet = max_street_bet
      if max_bet == 0:
        max_bet = 100
      good_string = "".join(bets)
      streets[i] = good_string
    return "/".join(streets), max_bet
