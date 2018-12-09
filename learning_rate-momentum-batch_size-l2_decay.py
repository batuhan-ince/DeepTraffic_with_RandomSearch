#lanesside,ahead,learnrate,batch degisiyor.Iteration 160000.
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pyperclip
import numpy as np
import random
import time

#Some requirements for login to DeepTraffic site with selenium
driver = webdriver.Chrome("B:\chromedriver") #You should download driver from selenium site and write the path of driver
driver.get("https://selfdrivingcars.mit.edu/login/")
user = "YourEmail"
pwd = "YourPassword"
elem = driver.find_element_by_id("user_login")
elem.send_keys(user)
elem = driver.find_element_by_id("user_pass")
elem.send_keys(pwd)
elem.send_keys(Keys.RETURN)
deeptraffic = driver.find_element_by_id("menu-item-327")
deeptraffic.click()
codebox = driver.find_element_by_xpath("//*[@id='container']/div/div[1]/textarea")
run_training = driver.find_element_by_id("trainButton")
start_test = driver.find_element_by_id("evalButton")
applycode = []
submit = []
applycode = driver.find_elements_by_class_name("button-small")[2]
submit = driver.find_elements_by_class_name("button-small")[5]


#Feel free to change hyperparameters
learning_rate = np.logspace(-3,0, num=6, dtype=float)
momentum = np.logspace(-3,0, num=6, dtype=float)
batch_size = np.logspace(4,8, num=6,base=2, dtype=int)
l2_decay = np.logspace(-3,0, num=6, dtype=float)

wait = WebDriverWait(driver, 216000)

best_concl = 0
best_hp1 = 0
best_hp2 = 0
best_hp3 = 0
best_hp4 = 0

for i in range(60):
    #Random select from logarithmic lists
    hp1 = random.choice(learning_rate)
    hp2 = random.choice(momentum)
    hp3 = random.choice(batch_size)
    hp4 = random.choice(l2_decay)

    #Javascript code from DeepLearning site
    #This 'f' before the text is needed for changing our hyperparameters. For more detail you can look "https://www.python.org/dev/peps/pep-0498/"
    text = f"""
    lanesSide = 4;
    patchesAhead = 30;
    patchesBehind = 10;
    trainIterations = 10000;
    otherAgents = 0; // max of 10
    var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
    var num_actions = 5;
    var temporal_window = 0;
    var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;
    var layer_defs = [];
    layer_defs.push({{
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
    }});
    layer_defs.push({{
    type: 'fc',
    num_neurons: 160,
    activation: 'tanh'
    }});
    layer_defs.push({{
    type: 'fc',
    num_neurons: 80,
    activation: 'tanh'
    }});
    layer_defs.push({{
    type: 'regression',
    num_neurons: num_actions
    }});
    var tdtrainer_options = {{
    learning_rate: {hp1},
    momentum: {hp2},
    batch_size: {hp3},
    l2_decay: {hp4}
    }};
    var opt = {{}};
    opt.temporal_window = temporal_window;
    opt.experience_size = 3000;
    opt.start_learn_threshold = 500;
    opt.gamma = 0.9;
    opt.learning_steps_total = 10000;
    opt.learning_steps_burnin = 1000;
    opt.epsilon_min = 0.0 ;
    opt.epsilon_test_time = 0.0;
    opt.layer_defs = layer_defs;
    opt.tdtrainer_options = tdtrainer_options;
    brain = new deepqlearn.Brain(num_inputs, num_actions, opt);
    learn = function (state, lastReward) {{
    brain.backward(lastReward);
    var action = brain.forward(state);
    draw_net();
    draw_stats();
    return action;
    }}
    """
    #These three lines should copy and paste avascript code to the site's console
    codebox.send_keys(Keys.CONTROL, "a")
    pyperclip.copy(text)
    codebox.send_keys(Keys.CONTROL, "v")

#All time.sleep functions are used for internet connection based errors.
    time.sleep(1)
    applycode.click()
    time.sleep(1)
    run_training.click()#Click Run Trainig Button
    wait.until(EC.visibility_of_element_located((By.XPATH,'/html/body/div[3]' )))#This provides us to wait the Run Training Process
    finished = driver.find_element_by_xpath('/html/body/div[3]/div[7]/div/button')
    finished.click()
    time.sleep(1)
    start_test.click()#Click Start Evaluation Test button
    wait.until(EC.visibility_of_element_located((By.XPATH, '/html/body/div[3]' )))#This provides us to wait the Start Evaluation Test Process
    time.sleep(1)

    #Try-Except used for Integer results
    try:
        concl = driver.find_element_by_xpath('/html/body/div[3]/p/b').text[0:5]
        concl = float(concl)
    except ValueError:
        concl = driver.find_element_by_xpath('/html/body/div[3]/p/b').text[0:2]
        concl = float(concl)

    finished.click()
    if best_concl < concl:
        best_concl = concl
        best_hp1 = hp1
        best_hp2 = hp2
        best_hp3 = hp3
        best_hp4 = hp4

        #Click and wait Submit Process.We selected 73, because our algorithm provides us to reach 73 mph border
    if concl >= 73:
        time.sleep(1)
        submit.click()
        wait.until(EC.visibility_of_element_located((By.XPATH, '/html/body/div[3]')))
        time.sleep(1)
        finished.click()
#Print Results
    print('learning rate: ', hp1, 'momentum: ', hp2, 'batch size: ', hp3, 'l2 decay: ', hp4, 'concl: ', concl, 'best_concl: ',best_concl)
print('best learning rate: ', best_hp1, 'momentum: ', best_hp2, 'batch size: ', best_hp3, 'l2 decay: ', best_hp4,'best_concl: ', best_concl)











