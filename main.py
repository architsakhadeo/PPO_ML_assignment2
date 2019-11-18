#!/usr/bin/env python3

"""
CMPUT 652, Fall 2019 - Assignment #2

__author__ = "Craig Sherstan"
__copyright__ = "Copyright 2019"
__credits__ = ["Craig Sherstan"]
__email__ = "sherstan@ualberta.ca"
"""

"""
You are free to additional imports as needed... except please do not add any additional packages or dependencies to
your virtualenv other than those specified in requirements.txt. If I can't run it using the virtualenv I specified,
without any additional installs, there will be a penalty.

I've included a number of imports that I think you'll need.
"""
import torch
import matplotlib
import matplotlib.pyplot as plt
import gym
from network import network_factory
from network import PolicyNetwork
from network import ValueNetwork
from torch.distributions import Categorical
import argparse
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter
import os
import sys
from torch import nn
from torch import optim
import pickle


# prevents type-3 fonts, which some conferences disallow.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def make_env():
    env = gym.make('CartPole-v0')
    return env


def sliding_window(data, N):
    """
    For each index, k, in data we average over the window from k-N-1 to k. The beginning handles incomplete buffers,
    that is it only takes the average over what has actually been seen.
    :param data: A numpy array, length M
    :param N: The length of the sliding window.
    :return: A numpy array, length M, containing smoothed averaging.
    """

    idx = 0
    window = np.zeros(N)
    smoothed = np.zeros(len(data))

    for i in range(len(data)):
        window[idx] = data[i]
        idx += 1

        smoothed[i] = window[0:idx].mean()

        if idx == N:
            window[0:-1] = window[1:]
            idx = N - 1

    return smoothed
    

def discount_returns(rewards, gamma=1):
    # Discounts rewards and stores their cumulative return in reverse
    '''
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r
    '''
    
    r = rewards[::-1] #rewards in reverse
    G = [r[0]] #last rewards
    for i in range(1,len(r)):
        G.append(r[i] + gamma*G[-1])
    G = G[::-1]
    G = np.array(G)
    return G
    

def lambda_returns(rewards, lambda_value_estimates, gamma = 0.99, l=0.95):
    r = rewards[::-1]
    V = lambda_value_estimates.view(1,-1)[0].tolist()[::-1]
    G = [r[0]]
    
    for i in range(1,len(r)):
        G.append(r[i] + gamma* ((1 - l)*V[i] + l*G[-1]) )
    G = G[::-1]
    return G


def reinforce(env, policy_estimator, value_estimator, num_episodes, # value_estimator=None,
              batch_size=2, gamma=1):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 0
    writer = SummaryWriter()
    
    # Define optimizer
    optimizer =   optim.Adam(policy_estimator.network.parameters(),  lr=0.00025)
    optimizer_v = optim.Adam(value_estimator.network_v.parameters(), lr=0.0001)
    
    action_space = np.arange(env.action_space.n)
    flag = 1     # 1 for train, 0 for test
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        
        while complete == False:
        
            # Gets reward and next state
            
            action = policy_estimator.get_action(s_0)
            s_1, r, complete, _ = env.step(action)
            
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # Checks if episode is over
                
            if complete:
                #print('States', len(states))
                batch_counter += 1
                #batch_rewards.extend(discount_returns(rewards, gamma))
                
                batch_states.extend(states)
                batch_actions.extend(actions)
                
                state_tensor = torch.tensor(batch_states, dtype=torch.float32)
                lambda_value_estimates = value_estimator.forward(state_tensor).detach()
                #lambda_returns(rewards, lambda_value_estimates, gamma=0.99, l=0.95)
                #print(discount_returns(rewards, gamma))
                batch_rewards.extend(lambda_returns(rewards, lambda_value_estimates, gamma=0.99, l=0.95))
                
                total_rewards.append(sum(rewards))
                
                #state_tensor = torch.tensor(batch_states, dtype=torch.float32)
                #reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
                #action_tensor = torch.tensor(batch_actions, dtype=torch.int32)
                                
                #prob_action_old = policy_estimator.forward(state_tensor)#.log_prob(action_tensor)
                
                # Updates after batch of episodes, here batch is 1
                
                if batch_counter == batch_size:
                    if flag == 1:
                        state_tensor = torch.tensor(batch_states, dtype=torch.float32)
                        prob_action_old = policy_estimator.forward(state_tensor).detach()
                        #value_estimates = value_estimator.forward(state_tensor)
                        #print(len(batch_rewards))
                        for epoch in range(10):
                            # Value update
                            indices = [i for i in range(len(batch_rewards))]
                            random.shuffle(indices)
                            for i in range(2):
                                miniindices = indices[int(1.0*i*len(batch_rewards)/2):int(1.0*(i+1)*len(batch_rewards)/2)]
                                minibatch_rewards = [batch_rewards[j] for j in miniindices]
                                minibatch_states = [batch_states[j] for j in miniindices]
                                minibatch_actions = [batch_actions[j] for j in miniindices]
                                                                                            
                                state_tensor_v = torch.tensor(minibatch_states, dtype=torch.float32)
                                reward_tensor_v = torch.tensor(minibatch_rewards, dtype=torch.float32)
                                value_estimates = value_estimator.forward(state_tensor_v)                                                              
                                loss_v = torch.mean((reward_tensor_v-value_estimates.view(1,-1)[0])**2)
                                optimizer_v.zero_grad()
                                loss_v.backward(retain_graph=True)
                                optimizer_v.step()                    
                                
                                # Policy update               
                                state_tensor = torch.tensor(minibatch_states, dtype=torch.float32)
                                reward_tensor = torch.tensor(minibatch_rewards, dtype=torch.float32)
                                action_tensor = torch.tensor(minibatch_actions, dtype=torch.int32)
                                
                                mini_prob_action_old = prob_action_old[miniindices].gather(1,action_tensor.long().view(-1,1))                                
                                prob_action_current = policy_estimator.forward(state_tensor).gather(1,action_tensor.long().view(-1,1))
                                
                                r = torch.div(prob_action_current,mini_prob_action_old)
                                clipped_r = torch.clamp(r, 1 - 0.2, 1 + 0.2)
  
                                final_r = torch.min(r, clipped_r)  #since min of -                                
                                advantage = reward_tensor-value_estimates.view(1,-1)[0]
                                normalized_advantage = (advantage - advantage.mean()) / advantage.std()
                                if len(advantage) <= 1:
                                    normalized_advantage = advantage/advantage    
                                #print('Normalized' , normalized_advantage)
                                #print(advantage)
                                #print
                                #print(advantage.mean())
                                #print
                                #print(advantage.std())
                                
                                loss = - torch.mean(1.0*final_r.view(1,-1)[0]*normalized_advantage) + loss_v + torch.mean( - 0.01 * Categorical(policy_estimator.forward(state_tensor)).entropy())
                                #loss = - torch.mean(1.0*torch.div(prob_action_current,mini_prob_action_old)*(reward_tensor-value_estimates[miniindices])) #-value_estimates
                                #loss = - torch.mean(Categorical(policy_estimator.forward(state_tensor)).log_prob(action_tensor)*(reward_tensor-value_estimates)) #-value_estimates
                                optimizer.zero_grad()   
                                loss.backward(retain_graph=True)                                          
                                optimizer.step()                                

                                '''
                                optimizer_v.zero_grad()                            
                                state_tensor_v = torch.tensor(batch_states, dtype=torch.float32)
                                reward_tensor_v = torch.tensor(batch_rewards, dtype=torch.float32)
                                value_estimates = value_estimator.forward(state_tensor_v)
                                loss_v = torch.mean((reward_tensor_v-value_estimates)**2)
                                loss_v.backward(retain_graph=True)
                                optimizer_v.zero_grad()
                                optimizer_v.step()                    
                                
                                # Policy update
                                optimizer.zero_grad()                            
                                state_tensor = torch.tensor(batch_states, dtype=torch.float32)
                                reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
                                action_tensor = torch.tensor(batch_actions, dtype=torch.int32)
                                loss = - torch.mean(policy_estimator.forward(state_tensor).log_prob(action_tensor)*(reward_tensor-value_estimates)) #-value_estimates
                                loss.backward()
                                optimizer.step()
                                '''            
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 0
                    
                print("Ep: {} Average of last 100: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-100:])))
                
                # Saves policy 
                
                #if (ep + 1) % 10000 == 0:
                #    torch.save(pe.network.state_dict(), 'saved_network_'+ str(ep + 1) + '_baseline.pkl')
                
                #if flag == 1:
                    
                    # Tensorboard plots
                    
                #    writer.add_scalar('return', total_rewards[-1], ep) #discounted rewards with gamma = 1, hence undiscounted
                #    writer.add_scalar('loss/policy', loss, ep)
                #    writer.add_scalar('loss/value', loss_v, ep)
                #    writer.add_scalar('loss/total', 0.98*loss + 0.02*loss_v, ep)
                #    for name, params in zip(policy_estimator.network.state_dict().keys(), policy_estimator.network.parameters()):
                #        average_grad = torch.mean(params.grad**2)
                #        writer.add_scalar('gradient_policy/'+str(name), average_grad, ep)
                #    for name, params in zip(value_estimator.network_v.state_dict().keys(), value_estimator.network_v.parameters()):
                #        average_grad = torch.mean(params.grad**2)
                #        writer.add_scalar('gradient_value/'+str(name), average_grad, ep)
                    
                
    return total_rewards

if __name__ == '__main__':

    """
    You are free to add additional command line arguments, but please ensure that the script will still run with:
    python main.py --episodes 10000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-e", default=5000, type=int, help="Number of episodes to train for")
    args = parser.parse_args()

    episodes = args.episodes

    """
    It is unlikely that the GPU will help in this instance (since the size of individual operations is small) - in fact 
    there's a good chance it could slow things down because we have to move data back and forth between CPU and GPU.
    Regardless I'm leaving this in here. For those of you with GPUs this will mean that you will need to move your 
    tensors to GPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numrun = 1
    
    for run in range(numrun):
        env = make_env()

        in_size = env.observation_space.shape[0]
        num_actions = env.action_space.n

        network = network_factory(in_size, num_actions, env)
        network.to(device)
        pe = PolicyNetwork(network)
        
        # Load policy to test
        #pe.network.load_state_dict(torch.load('saved_network_50000_baseline.pkl'))
        
        ve = ValueNetwork(in_size)
        ep_returns = reinforce(env, pe, ve, episodes) #,ve , loss_policy, loss_value
            
        #fwrite = open('runs_data/'+str(run)+'.pkl','wb')
        #fwrite = open('runs_data/0.pkl','wb')
        #pickle.dump(ep_returns, fwrite)
        #fwrite.close()
            
        
        
    window = 100
    plt.figure(figsize=(12,8))
    plt.plot(sliding_window(ep_returns, window))
    plt.title("Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Average Return (Sliding Window 100)")
    plt.show()

    # save your network
    #torch.save(pe.network.state_dict(), 'saved_network_50000_baseline.pkl')
    
