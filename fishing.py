import numpy as np
import gymnasium as gym
from gymnasium import spaces

_DEFAULT_INIT_POP = np.array([0.7,0.6,1], dtype=np.float32)

def penalty(t):
  return - 100/(t+1)


def pop_growth(pop, parameters, *args, **kwargs):
  X, Y, Z = pop[0], pop[1], pop[2]
  p = parameters
    
  coupling = p["v0"]**2 
  K_x = p["K_x"]

  pop[0] += (p["r_x"] * X * (1 - X / K_x)
        - p["beta"] * Z * (X**2) / (coupling + X**2)
        - p["cV"] * X * Y
        + p["tau_yx"] * Y - p["tau_xy"] * X  
        + p["sigma_x"] * X * np.random.normal()
       )
    
  pop[1] += (p["r_y"] * Y * (1 - Y / p["K_y"] )
        - p["D"] * p["beta"] * Z * (Y**2) / (coupling + Y**2)
        - p["cV"] * X * Y
        - p["tau_yx"] * Y + p["tau_xy"] * X  
        + p["sigma_y"] * Y * np.random.normal()
       )

  pop[2] += p["alpha"] * (
      Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
      + p["sigma_z"] * Z  * np.random.normal()
    )        
    
  # consider adding the handling-time component here too instead of these   
  #Z = Z + p["alpha"] * (Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
  #                      + p["sigma_z"] * Z  * np.random.normal())
                          
  pop = pop.astype(np.float32)
  return(pop)

class fishing_env(gym.Env):
  """3 species, 2 are fished"""
  def __init__(self, config=None):
    self.config = config or {}
    #
    self.parameters = config.get("parameters", None)
    self.growth_fn = config.get(
      "growth_fn", 
      pop_growth
      )
    #
    self.initial_pop = config.get("initial_pop", _DEFAULT_INIT_POP)
    self.Tmax = config.get("Tmax", 200)
    self.threshold = config.get("threshold", np.float32(5e-2))
    self.init_sigma = config.get("init_sigma", np.float32(5e-3))
    #
    self.cost = config.get("cost", np.float32([0.0, 0.0]))
    self.relative_weights = config.get("relative_weights", np.float32([1,1]))
    self.bound = self.config.get("bound", 4)
    # self.early_end_penalty = self.config.get("early_end_penalty", default_penalty)
    self.timestep = 0
    #
    self.action_space = spaces.Box(
      np.array([-1, -1], dtype=np.float32),
      np.array([1, 1], dtype=np.float32),
      dtype = np.float32
    )
    self.observation_space = spaces.Box(
      np.array([1, -1, -1], dtype=np.float32),
      np.array([1, 1, 1], dtype=np.float32),
      dtype=np.float32,
    )
    self.reset(seed=config.get("seed", None))
    #################################################################
  
  def reset(self, *, seed = None, options = None):
    # "*" forces keyword args to be named on calls if values are provided
    self.timestep = 0
    self.state = self.get_state(self.initial_pop)
    self.state += np.float32(self.init_sigma * np.random.normal(size=3) )
    info = {}
    return self.state, info
  
  def step(self, action):
    action = np.clip(action, [0, 0], [1, 1])
    # action *= 0.5
    pop = self.get_population() # current state in natural units
    
    # harvest & recruitment
    pop, reward = self.harvest(pop, action)
    pop = self.growth_fn(pop, self.parameters, self.timestep)
    
    # Conservation goals:
    terminated = False
    if any(pop <= self.threshold):
      terminated = True
      # reward += self.early_end_penalty(self.timestep)
            
    self.state = self.get_state(pop) # transform into [0, 1] space
    observation = self.state
    self.timestep += 1
    return observation, reward, terminated, False, {}
  
  ## Functions beyond those standard for gym.Env's:
  
  def harvest(self, pop, action):
    harvested_fish = pop[:-1] * (action+1)/2
    pop[0] = max(pop[0] - harvested_fish[0], 0)
    pop[1] = max(pop[1] - harvested_fish[1], 0)
    reward_vec = (
      harvested_fish * self.relative_weights - self.cost * action
    )
    total_reward = sum(reward_vec)
    return pop, np.float32(total_reward)
  
  def get_state(self, pop):
    return 2 * pop / self.bound - 1

  def get_population(self):
    return (self.state + 1) * self.bound / 2
  
