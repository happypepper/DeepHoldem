from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import os
import sys
import time
import slum_util
import socket

if len(sys.argv) < 3:
  print("missing address port")
  sys.exit()
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((sys.argv[1], int(sys.argv[2])))

chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.binary_location = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary'

d = DesiredCapabilities.CHROME
d['loggingPrefs'] = { 'browser':'ALL' }

driver = webdriver.Chrome(executable_path="/Users/lawson/Downloads/chromedriver", chrome_options=chrome_options, desired_capabilities=d)
driver.get("http://slumbot.com")

time.sleep(2)

response_fun = """
response = function(data) {
    global_data = data
    // We increment actionindex even when we get an error message back.
    // If we didn't, then when we retried the action that triggered the error,
    // it would get flagged as a duplicate.
    ++actionindex;
    if ("errormsg" in data) {
	var errormsg = data["errormsg"];
	$("#msg").text(errormsg);
	// Some errors end the hand (e.g., server timeout)
	// Would it be cleaner to treat a server timeout like a client
	// timeout?  Return msg rather than errormsg?
	if ("hip" in data) {
	    handinprogress = (data["hip"] === 1);
	}
	// Need this for server timeout.  Want to enable the "Next Hand"
	// button and disable all the other buttons.
	enableactions();
	return;
    } else if ("msg" in data) {
	var msg = data["msg"];
	$("#msg").text(msg);

    } else {
	$("#msg").text("");
    }

    if (actiontype === 1) {
	addourcheck();
    } else if (actiontype === 2) {
	addourcall();
    } else if (actiontype === 3) {
	addourfold();
    } else if (actiontype === 4) {
	addourbet();
    }
    $("#betsize").val("");
    potsize = data["ps"];
    ourbet = data["ourb"];
    oppbet = data["oppb"];
    var lastcurrentaction = currentaction;
    currentaction = data["action"];
    var actiondisplay = currentaction;
    $("#currentaction").text(actiondisplay);
    var overlap = currentaction.substring(0, lastcurrentaction.length);
    if (overlap !== lastcurrentaction) {
	console.log("Overlap " + overlap);
	console.log("Last current action " + lastcurrentaction);
    } else {
	var newaction = currentaction.substring(lastcurrentaction.length,
						currentaction.length);
	oppactionmessage(newaction);
    }

    parsedata(data);
    drawall(aftershowdown);
    lifetimetotal = data["ltotal"];
    lifetimeconf = data["lconf"];
    lifetimebaselinetotal = data["lbtotal"];
    lifetimebaselineconf = data["lbconf"];
    numlifetimehands = data["lhands"];
    showdowntotal = data["sdtotal"];
    showdownconf = data["sdconf"];
    numshowdownhands = data["sdhands"];
    blbshowdowntotal = data["blbsdtotal"];
    blbshowdownconf = data["blbsdconf"];
    blbnumshowdownhands = data["blbsdhands"];
    clbshowdowntotal = data["clbsdtotal"];
    clbshowdownconf = data["clbsdconf"];
    clbnumshowdownhands = data["clbsdhands"];
    if (username !== "") displaystats();
    if (! handinprogress) {
	sessiontotal = data["stotal"];
	$("#sessiontotal").text(sessiontotal);
	numsessionhands = data["shands"];
	$("#numhands").text(numsessionhands);
	var outcome = data["outcome"];
	if (outcome > 0) {
	    $("#outcome").text("You won a pot of " + outcome + "!");
	} else if (outcome < 0) {
	    $("#outcome").text("Slumbot won a pot of " + -outcome +
			       "!");
	} else {
            $("#outcome").text("You chopped!");
	}
    } else {
	$("#outcome").text("");
	starttimer();
    }
    enableactions();
};
"""
driver.execute_script(response_fun)
hand_no = 0

while True:
  nexthand_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "nexthand")))
  nexthand_button.click()
  hand_no += 1

  position = 0
  while True:
    fold_button = driver.find_element_by_id("fold")
    nexthand_button = driver.find_element_by_id("nexthand")
    action_td = driver.find_element_by_id("currentaction")

    if fold_button.is_displayed() and fold_button.is_enabled():
      if action_td.text:
        position = 1
      break
    if nexthand_button.is_displayed() and nexthand_button.is_enabled():
      break
    time.sleep(1)

  # new hand
  while True:
    nexthand_button = driver.find_element_by_id("nexthand")
    if nexthand_button.is_displayed() and nexthand_button.is_enabled():
      break
    hole = driver.execute_script("return global_data[\"holes\"]")
    actions = driver.execute_script("return global_data[\"action\"]")
    board = driver.execute_script("return global_data[\"board\"]")

    fold_button = driver.find_element_by_id("fold")
    call_button = driver.find_element_by_id("call")
    check_button = driver.find_element_by_id("check")
    halfpot_button = driver.find_element_by_id("halfpot")
    pot_button = driver.find_element_by_id("pot")
    allin_button = driver.find_element_by_id("allin")

    while True:
      if (call_button.is_displayed() and call_button.is_enabled()) or (allin_button.is_displayed() and allin_button.is_enabled()):
        break
      time.sleep(1)

    # ready to make the action
    actions, max_bet = slum_util.acpcify_actions(actions)
    msg = "MATCHSTATE:" + str(position) + ":" + str(hand_no) +":" + actions + ":"
    if position == 0:
      msg += hole + "|"
    elif position == 1:
      msg += "|" + hole
    if len(board) > 0:
      msg += "/" + slum_util.acpcify_board(board)
    msg += "\n"

    client_socket.send(msg)
    sys.stdout.write("sent " + msg + ":")
    advice = client_socket.recv(100)

    # click button
    print(advice)

    if advice == "c":
      if call_button.is_displayed() and call_button.is_enabled():
        call_button.click()
      elif check_button.is_displayed() and check_button.is_enabled():
        check_button.click()
    elif advice == "f":
      fold_button.click()
    elif advice == "20000":
      allin_button.click()
    elif advice == str(max_bet*3):
      pot_button.click()
    else:
      halfpot_button.click()

    while True:
      time.sleep(1)
      if (call_button.is_displayed() and call_button.is_enabled()) or (allin_button.is_displayed() and allin_button.is_enabled()):
        break
      if nexthand_button.is_displayed() and nexthand_button.is_enabled():
        break

#signup_button = driver.find_element_by_id("login_trigger")
#signup_button.click()
