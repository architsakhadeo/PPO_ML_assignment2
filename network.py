"""
CMPUT 652, Fall 2019 - Assignment #2

__author__ = "Craig Sherstan"
__copyright__ = "Copyright 2019"
__credits__ = ["Craig Sherstan"]
__email__ = "sherstan@ualberta.ca"
"""

import torch
from torch import nn
import numpy as np
from torch.distributions import Categorical

def network_factory(in_size, num_actions, env):
    """

    :param in_size:
    :param num_actions:
    :param env: The gym environment. You shouldn't need this, but it's included regardless.
    :return: A network derived from nn.Module
    """
    network = nn.Sequential(nn.Linear(in_size, 32), nn.ReLU(), nn.Linear(32, num_actions), nn.Softmax(dim=-1))
    return network

    
class PolicyNetwork(nn.Module):
    def __init__(self, network):
        super(PolicyNetwork, self).__init__()
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action_probs = self.network(state)
        #return Categorical(action_probs)
        #print('Action probs ', action_probs)
        return(action_probs)

    def get_action(self, state):
        """
        This function will be used to evaluate your policy.
        :param inputs: environmental inputs. These should be the environment observation wrapped in a tensor:
        torch.tensor(obs, device=device, dtype=torch.float32)
        :return: Should return a single integer specifying the action
        """
        category = Categorical(self.forward(state))
        return category.sample().item()


class ValueNetwork(nn.Module):
    def __init__(self, in_size):
        super(ValueNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_v = nn.Sequential(nn.Linear(in_size, 32, bias=True), nn.ReLU(), nn.Linear(32, 1, bias=True))
        
    def forward(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        return self.network_v(state)

    def get_value(self, state):
        """
        This function will be used to evaluate your policy.
        :param inputs: environmental inputs. These should be the environment observation wrapped in a tensor:
        torch.tensor(obs, device=device, dtype=torch.float32)
        :return: Should return a single integer specifying the action
        """
        return self.forward(state).item()
