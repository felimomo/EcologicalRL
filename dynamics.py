import numpy as np


def dynamics(pop, params):
	return (
		pop 
		+ params['r'] * pop * (1 - pop / params['k'])
		- params['beta'] * params['h'] * (pop ** 2) / (params['c'] ** 2 + pop ** 2)
	)

def simulate_dynamics(T, params, init_pop):
	pop = init_pop
	sim = {'t': [0], 'pop': [pop]}
	for t in range(T):
		pop = dynamics(pop, params)
		sim['t'].append(t+1)
		sim['pop'].append(pop)
	return sim