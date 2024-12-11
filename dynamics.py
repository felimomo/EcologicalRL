import numpy as np


def dynamics(pop, params):
  #
  deterministic_change = (
    params['r'] * pop * (1 - pop / params['k'])
    - params['beta'] * params['h'] * (pop ** 2) / (params['c'] ** 2 + pop ** 2)
  )
  randomness = (1 + np.random.normal() * params['sigma'])
  change = deterministic_change * randomness
  #
  return pop + change

def simulate_dynamics(T, params, init_pop):
	pop = init_pop
	sim = {'t': [0], 'pop': [pop]}
	for t in range(T):
		pop = dynamics(pop, params)
		sim['t'].append(t+1)
		sim['pop'].append(pop)
	return sim