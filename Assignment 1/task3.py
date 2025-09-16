"""
Task 3: Optimized KL-UCB Implementation

This file implements both standard and optimized KL-UCB algorithms for multi-armed bandits.
The optimized version aims to reduce computational overhead while maintaining good regret performance.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Base Algorithm Class ------------------

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# ------------------ KL-UCB utilities ------------------
## You can define other helper functions here if needed

def kl_div(p, q):
    p = min(max(p,1e-10),1-1e-10)
    q = min(max(q,1e-10),1-1e-10)
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))


def find_ucb(mean, pulls, t, c, max_it):
    val = (math.log(t)+c*math.log(max(math.log(max(t,2)),1.0000001)))/pulls
    low = mean
    high = 1.0
    for _ in range (max_it) :
        mid = low + (high-low)/2.0
        if kl_div(mean,mid)<=val:
            low = mid
        else:
            high = mid
        
    return 0.5*(high+low)
# ------------------ Optimized KL-UCB Algorithm ------------------

class KL_UCB_Optimized(Algorithm):
    """
    Optimized KL-UCB algorithm that reduces computation while maintaining identical regret.
    This implements a batched KL-UCB with exponential+binary search for safe pulls of the current best arm.
    """
    # You can define other functions also in the class if needed
    
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        self.counts = np.zeros(self.num_arms)
        self.sums = np.zeros(self.num_arms)
        self.t = 0
        self.current = None
        self.remaining = 0
        self.multiplier = 50
        self.max_batch = horizon
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        self.t += 1

        if self.remaining > 0:
            self.remaining -= 1
            return self.current
        
        for a in range(self.num_arms):
            if self.counts[a] == 0:
                self.current = a
                return a

        means = self.sums/self.counts
        ucbs = [find_ucb(means[a],self.counts[a],self.t,c=3,max_it =5) for a in range(self.num_arms)]
        order =np.argsort(ucbs)[::-1]
        best,second = order[0],order[1]
        gap = max(ucbs[best]-ucbs[second],0.0)

        batch=max(1,min(self.max_batch,self.multiplier*gap))
        if gap<0.1: 
            batch=1

        self.current = best
        self.remaining = batch-1
        return best
        #END EDITING HERE


    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        self.counts[arm_index] += 1
        self.sums[arm_index]   += reward
        #END EDITING HERE

# ------------------ Bonus KL-UCB Algorithm (Optional - 1 bonus mark) ------------------

class KL_UCB_Bonus(Algorithm):
    """
    BONUS ALGORITHM (Optional - 1 bonus mark)
    
    This algorithm must produce EXACTLY IDENTICAL regret trajectories to KL_UCB_Standard
    while achieving significant speedup. Students implementing this will earn 1 bonus mark.
    
    Requirements for bonus:
    - Must produce identical regret trajectories (checked with strict tolerance)
    - Must achieve specified speedup thresholds on bonus testcases
    - Must include detailed explanation in report
    """
    # You can define other functions also in the class if needed

    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        pass
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        pass
        #END EDITING HERE
