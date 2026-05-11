#ensures compatibility between python2 and python3 
from __future__ import absolute_import 
from __future__ import print_function

import os
import sys
import time  # to measure code execution time
import optparse #to parse command line options, for processing arguments passed to the script when it is executed
import random
import serial  #for serial communication with external devices
import numpy as np
import torch
import torch.optim as optim    #contains commonly used optimizers to adjust model learning parameters
import torch.nn.functional as F # contains activation functions and other functions used in model definition
import torch.nn as nn  # for the construction of neural networks
import matplotlib.pyplot as plt

# connexion to sumo
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  
import traci  # control simulator elements(vehicles, traffic lights)

def get_vehicle_numbers(lanes):
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:   #Check if the position of the vehicle (k) on the track is more than 10 meters to  filter vehicles that have already advanced a certain distance
                vehicle_per_lane[l] += 1
    return vehicle_per_lane #a dictionary that contains the number of vehicle in each lane


def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)  #traci.lane.getWaitingTime(lane) returns the wait time on the specified lane.
    return waiting_time    #Returns the total sum of wait time on all lanes


def phaseDuration(junction, phase_time, phase_state): #junction: the identifier of the crossroads (traffic light) that we want to control.| phase_time: the duration (in seconds) that we want to assign to the specific phase of the fire.| phase_state: the state (red, yellow, green) you want to assign to the phase of the light.
    traci.trafficlight.setRedYellowGreenState(junction, phase_state) # give a state to the junction (red,green,yellow)
    traci.trafficlight.setPhaseDuration(junction, phase_time) # give a duration to that state of the junction



class Model(nn.Module):  # this class is intended to be used as a neural network model in PyTorch
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions): # the constructure of the function : lr: learning rate, fc1_dims: Number of neurons in the first fully connected layer (fully connected), n_actions : Number of possible actions the model can predict
        super(Model, self).__init__() #Calls the constructor of the parent class (nn.Module) to initialize some basic class features.
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims) #Declares the first linear layer (int, out)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr) #Initializes an Adam optimizer that will be used to adjust the model parameters, self.parameters() returns all model parameters,
        self.loss = nn.MSELoss() # initialize the loos function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Determines whether a GPU (CUDA) is available and configures the device (device) accordingly
        self.to(self.device) #Moves all model parameters to the specified device (CPU or GPU)

    def forward(self, state): #specifies how the data passes through the model, It takes an input state and returns the model predictions for that state.
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions


class Agent:
    def __init__( # constructure 
        self,
        gamma, #discount factor in future reward function
        epsilon, # initial scan rate
        lr, # learning rate
        input_dims, 
        fc1_dims, #number of neurones in layer 1 (fc: fully connected)
        fc2_dims, ##number of neurones in layer 2 
        batch_size,
        n_actions,
        junctions, #intersection identifiers (traffic lights)
        max_memory_size=100000,
        epsilon_dec=5e-4, #decrease in exploration rate
        epsilon_end=0.05, #minimum scan rate value.
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = Model(
            self.lr, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions
        )
        self.memory = dict()
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "new_state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "reward_memory":np.zeros(self.max_mem, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem, dtype=bool),
                "mem_cntr": 0,
                "iter_cntr": 0,
            }


    def store_transition(self, state, state_, action,reward, done,junction): #Stores a transition in agent memory 
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = state_
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation): #Chooses an action based on the current observation. Uses the Q model to make exploration-based decisions (epsilon-greedy).
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device) #Converts the observation to a PyTorch tensor (torch.tensor) and sends it to the device (device) used by the Q model (self.Q_eval). This prepares the observation to be used as input for model Q
        #epsilon-greedy:
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action #return the choosen action
    
    def reset(self,junction_numbers): #allows to reset the agent memory
        for junction_number in junction_numbers:
            self.memory[junction_number]['mem_cntr'] = 0

    def save(self,model_name): #save the weights of model Q in a binary file
        torch.save(self.Q_eval.state_dict(),f'models/{model_name}.bin')

    def learn(self, junction): #implements the Q-model learning algorithm with the Q-learning method
        self.Q_eval.optimizer.zero_grad() #Resets the optimizer gradients to avoid accumulation of gradients from previous training steps.

        batch= np.arange(self.memory[junction]['mem_cntr'], dtype=np.int32) #Creates a NumPy array containing indices from 0 to mem_cntr - 1 to form a batch with the data stored in the specified hub memory.

        state_batch = torch.tensor(self.memory[junction]["state_memory"][batch]).to(
            self.Q_eval.device
        )#Creates a batch of PyTorch tensors from the current states stored in the Hub memory.
        new_state_batch = torch.tensor(
            self.memory[junction]["new_state_memory"][batch]
        ).to(self.Q_eval.device)#Creates a batch of PyTorch tensors from the new states stored in the Hub memory.
        reward_batch = torch.tensor(
            self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device) #Creates a batch of PyTorch tensors from the rewards stored in the Hub’s memory.
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device) #Creates a batch of PyTorch tensors from the termination indicators stored in the hub memory.
        action_batch = self.memory[junction]["action_memory"][batch] #Retrieves the actions stored in the hub memory for the batch.

        q_eval = self.Q_eval.forward(state_batch)[batch, action_batch] #Passes the current state batch through the Q model to obtain the predicted Q values for the actions taken.
        q_next = self.Q_eval.forward(new_state_batch) #Passes the batch of new states through the Q model to obtain the predicted Q values for the following actions.
        q_next[terminal_batch] = 0.0 #Zeroes the predicted Q values for the following states if they are terminal.
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0] #Calculates the Q target for learning by combining current rewards with future reward estimates weighted by the discount factor (gamma)
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device) #Calculates the loss (error) between the target Q values and the predicted Q values using the loss function specified in the Q model.

        loss.backward() #Retroactively propagates gradients to calculate derivatives relative to model weights.
        self.Q_eval.optimizer.step() #Performs an optimization step to adjust the model weights according to the calculated gradients.

        self.iter_cntr += 1 #Increments the iteration counter, indicating the total number of learning iterations performed.
        #Updates the exploration rate with an exponential decrease.
        self.epsilon = (
            self.epsilon - self.epsilon_dec
            if self.epsilon > self.epsilon_end
            else self.epsilon_end
        )


def run(train=True,model_name="model",epochs=15,steps=200):
    #initialisation
    epochs = epochs
    steps = steps
    best_time = np.inf #Sets (best_time) with infinite value
    total_time_list = list() #Sets (total_time_list) with an empty list to store total times by time
    total_reward_list = list()
    #Starts the SUMO simulation according to the mode (train or evaluation) with the specified options, including the SUMO configuration file, the tripinfo.xml output file and, if applicable, the GUI mode.
    traci.start(
        [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"]
    ) 

    #Retrieves the list of all hubs in the simulation and creates a list of clues for those hubs.
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))

    #Initializes an object in the Agent class with specified parameters, including agent characteristics, hyperparameters, and the list of intersections.
    brain = Agent(
        gamma=0.99,
        epsilon=0.0,
        lr=0.1,
        input_dims=4,
        # input_dims = len(all_junctions) * 4,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=1024,
        n_actions=4,
        junctions=junction_numbers,
    )

    if not train: #If in evaluation mode, loads the weights of the Q model from the specified file.
        brain.Q_eval.load_state_dict(torch.load(f'models/{model_name}.bin',map_location=brain.Q_eval.device))

    #Displays the device (CPU or GPU) on which the Q model is run and closes the SUMO simulation.
    print(brain.Q_eval.device)
    traci.close()
    for e in range(epochs): #Loop on the specified number of periods.
        if train: #If you are in training mode, restart the SUMO simulation with the different output file.
            traci.start(
            [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            ) 
        else:
            traci.start(
            [checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )

        print(f"epoch: {e}") #Affiche le numéro de l'époque actuelle.
        #Defines a select_lane matrix to specify the signal light phases for each possible action.
        select_lane = [
            ["rggg", "yggg"],
            ["yggg", "rggg"],
            ["gggg", "yggg"],
            ["yggg", "gggg"],
        ]

       
        #initialisation
        step = 0
        total_time = 0
        min_duration = 5
        
        traffic_lights_time = dict()
        prev_wait_time = dict()
        prev_vehicles_per_lane = dict()
        prev_action = dict()
        all_lanes = list()
        
        #Initializes dictionary values for each hub.
        for junction_number, junction in enumerate(all_junctions):
            prev_wait_time[junction] = 0
            prev_action[junction_number] = 0
            traffic_lights_time[junction] = 0
            prev_vehicles_per_lane[junction_number] = [0] * 4
            # prev_vehicles_per_lane[junction_number] = [0] * (len(all_junctions) * 4) 
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        while step <= steps: #Main loop that simulates each step (no time) of the simulation.
            traci.simulationStep()  #Moves to the next simulation step.
            for junction_number, junction in enumerate(all_junctions): #Loop on each crossroads.
                #Gets the lanes controlled by the intersection and calculates the total waiting time on those lanes.
                controled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controled_lanes)
                total_time += waiting_time #Adds current wait time to total time
                if traffic_lights_time[junction] == 0: #Check if the time of the current phase of the lights has elapsed.
                    vehicles_per_lane = get_vehicle_numbers(controled_lanes)
                    # vehicles_per_lane = get_vehicle_numbers(all_lanes)

                    #storing previous state and current state
                    reward = -1 *  waiting_time #Gets the number of vehicles per lane and calculates the reward as the opposite of the waiting time.
                    
                    state_ = list(vehicles_per_lane.values()) 
                    #Prepares the current and previous status as lists of vehicle numbers.
                    state = prev_vehicles_per_lane[junction_number]
                    prev_vehicles_per_lane[junction_number] = state_
                    brain.store_transition(state, state_, prev_action[junction_number],reward,(step==steps),junction_number) #Stores the transition (previous state, current state, previous action, reward, termination indicator) in the agent’s memory.

                    #selecting new action based on current state
                    lane = brain.choose_action(state_)
                    prev_action[junction_number] = lane
                    phaseDuration(junction, 6, select_lane[lane][0])
                    phaseDuration(junction, min_duration + 10, select_lane[lane][1])

                    #Initialize time for the next phase of fires.
                    traffic_lights_time[junction] = min_duration + 10
                    if train: #If you are in training mode, perform a learning step using the agent’s learn function with the crossroads index.
                        brain.learn(junction_number)
                else: #If in evaluation mode, decrement the remaining time for the fire phase.
                    traffic_lights_time[junction] -= 1
            step += 1 #Increments the step counter.
        print("total_time",total_time) #Displays the total waiting time for the time.
        total_time_list.append(total_time) #Adds the total wait time to the list for follow-up.
        total_reward_list.append(reward)
        

        if total_time < best_time: #If the current total wait time is the best so far, save the weights of model Q.
            best_time = total_time
            if train:
                brain.save(model_name)

        traci.close() #close SUMO simulation
        sys.stdout.flush() #Force l’évacuation du tampon de la sortie standard.
        if not train: #If one is in evaluation mode, goes out of the loop of times.
            break
    if train:#If you are in training mode, trace the curve of the evolution of the total waiting time over time.
        plt.plot(list(range(len(total_time_list))),total_time_list)
        plt.xlabel("epochs")
        plt.ylabel("total time")
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        plt.show()
    if train:
    # Affichage du graphe des récompenses
        plt.plot(list(range(len(total_reward_list))), total_reward_list)
        plt.xlabel("epochs")
        plt.ylabel("score")
        plt.savefig(f'plots/score_vs_epoch_{model_name}.png')
        plt.show()


# retrieve command line options and arguments
def get_options(): 
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="name of model",
    )
    optParser.add_option(
        "--train",
        action = 'store_true',
        default=False,
        help="training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps",
    )
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    run(train=train,model_name=model_name,epochs=epochs,steps=steps)
