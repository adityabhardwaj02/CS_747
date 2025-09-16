"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def kl_divergence(p,q):
    eps=1e-10
    p = np.clip(p, a_max=1-eps, a_min = eps)
    q = np.clip(q, a_max = 1-eps, a_min = eps)
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))

def find_q(arm_index, t, ut,p, c, eps,max_it):
    # val = (math.log(t)+c*math.log(math.log(t)))/ut
    val = (math.log(t)+c*math.log(max(math.log(max(t,2)),1.0000001)))/ut
    low = p
    high = 1.0
    for _ in range (max_it) :
        mid = low + (high-low)/2
        if kl_divergence(p,mid)<=val:
            low = mid
        else:
            high = mid
        
    return 0.5*(high+low)

    
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.ucb = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if np.sum(self.counts)<self.num_arms:
            arm_indx = int(np.sum(self.counts))
            return arm_indx
        else:
            return np.argmax(self.ucb)
        # END EDITING HERE  

        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        n=self.counts[arm_index]
        value=self.values[arm_index]
        new_value= ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index]=new_value
        t=np.sum(self.counts)
        self.ucb = self.values + np.sqrt(2*math.log(t)/(self.counts+1e-7))
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts=np.zeros(num_arms)
        self.values=np.zeros(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if np.sum(self.counts)<self.num_arms:
            arm_indx = int(np.sum(self.counts))
            return arm_indx
        else:
            optimal_arm=0
            q_max=0
            for i in range (self.num_arms):
                q = find_q(i, t= int(np.sum(self.counts)), ut= self.counts[i], p = self.values[i], c=2, eps=1e-6, max_it =10)
                if q>q_max:
                    q_max = q 
                    optimal_arm=i
            return optimal_arm
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        n=self.counts[arm_index]
        value=self.values[arm_index]
        new_value= ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index]=new_value

        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        
        self.success=np.zeros(self.num_arms)
        self.failures=np.zeros(self.num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # Sample from Beta distribution for each arm
        beta = np.random.beta(self.success+1, self.failures+1)
        return np.argmax(beta)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward==0:
            self.failures[arm_index]+=1
        else:
            self.success[arm_index]+=1
        # END EDITING HERE
    


