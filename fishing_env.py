import numpy as np
import gymnasium as gym
from gymnasium import spaces

import dynamics as dyn

class FishingEnv(gym.Env):
  def __init__(self, config={}):
    self.parameters = {}
    
    # logistic growth
    self.parameters['r'] = config.get('r', 0.4) 
    self.parameters['k'] = config.get('k', 1) 

    # predation by constant predator pop
    self.parameters['h'] = config.get('h', 0.1) 
    self.parameters['beta'] = config.get('beta', 1)
    self.parameters['c'] = config.get('c', 0.3) 

    # randomness
    self.parameters['sigma'] = config.get('sigma', 1.0)

    # initialization
    self.init_pop = np.array([0.5], dtype=np.float32)

    # bounds
    self.max_pop = 3 * self.parameters['k']
    self.max_t = config.get('max_t', 100)

    # RL algos usually 'tuned' to deal well with [-1, 1] spaces
    self.observation_space = spaces.Box(
      np.array([-1], dtype=np.float32),
      np.array([1], dtype=np.float32),
      dtype = np.float32,
    )
    self.action_space = spaces.Box(
      np.array([-1], dtype=np.float32),
      np.array([1], dtype=np.float32),
      dtype = np.float32,
    )

  def reset(self, seed=None, options=None):
    self.pop = (
      self.init_pop 
      * (1 + self.parameters['sigma'] * np.random.normal())
    )
    self.pop = np.clip(self.pop, 0, self.max_pop)
    #
    self.t=0
    #
    obs = self.observe()
    info = {}
    return obs, info

  def step(self, action):
    # might be trivial-looking but is crucial!
    np.clip(action, -1, 1)
    #
    harvest_rate = self.compute_harvest(action)
    harvested_pop = harvest_rate * self.pop
    #
    self.pop -= harvested_pop # harvest
    self.pop = dyn.dynamics(pop=self.pop, params=self.parameters) # recruit
    self.t += 1
    #
    reward = harvested_pop # could consider other rewards here
    obs = self.observe()
    truncated = False # not super relevant
    if self.t > self.max_t:
      terminated = True
    else:
      terminated = False
    #
    info = {}
    return obs, reward, terminated, truncated, info

  def observe(self):
    # pop: [0, max_pop] --> state: [-1, +1]
    return 2 * self.pop / self.max_pop - 1

  def compute_harvest(self, action):
    # action: [-1, 1] --> harvest rate: [0, 1] 
    return (action[0] + 1) / 2