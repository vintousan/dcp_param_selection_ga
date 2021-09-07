#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Genetic Algorithm-based Selection of Dark Channel Prior (DCP) Parameters for Underwater Image Restoration
# Description: A simple GA code that provides a combination of DCP parameters with maximum entropy
# Author: VJA


# In[2]:


# Import library and modules
import random
import numpy
from deap import base, creator, tools, algorithms


# In[3]:


# USER-DEFINED PARAMETERS
# Population, number of individuals per generation
POPULATION = 50
# Generation, number of times an evolution happens
GENERATION = 50
# Crossover Rate
CXPB = 0.5
# Mutation Rate
MUTPB = 0.01


# In[4]:


# Define fitness function
def fitness_fcn(individual):
    
    omega = individual[0]
    t_zero = individual[1]
    
    # Fitness Function
    # Relationship between DCP parameters and entropy generated through multivariate polynomial regression
    entropy = 7.4717 - (0.3690 * omega) - (0.4773 * t_zero) - (1.1122 * (omega ** 2)) - (0.1799 * (t_zero ** 2))  + (1.7033 * omega * t_zero)
    
    return entropy,


# In[5]:


# Check chromosome constraints 
def feasibility(individual):

    omega = individual[0]
    t_zero = individual[1]
    
    if (0 <= omega <= 1) and (0.1 <= omega < 0.95):
        return True
    return False


# In[6]:


# Extract current population statistics for Gaussian-based mutation
def gaussian_param(current_population):
    
    sum_omega = 0
    sum_t_zero = 0
    
    # Mean
    for ind in current_population:
        sum_omega += ind[0]
        sum_t_zero += ind[1]
        
    avg_omega = float(sum_omega/len(current_population))
    avg_t_zero = float(sum_t_zero/len(current_population))
    
    sum_omega = 0
    sum_t_zero = 0
    
    # Standard Deviation
    for ind in current_population:
        sum_omega = ((ind[0] - avg_omega) ** 2)    
        sum_t_zero = ((ind[0] - avg_t_zero) ** 2)    
        
    var_omega =  float(sum_omega/(len(current_population) - 1))
    var_t_zero =  float(sum_t_zero/(len(current_population) - 1))
    
    from math import sqrt
    
    sd_omega = sqrt(var_omega)
    sd_t_zero = sqrt(var_t_zero)
    
    return avg_omega, sd_omega, avg_t_zero, sd_t_zero


# In[7]:


def set_fitness(population):

    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        
    fits = [ind.fitness.values[0] for ind in population]
    
    return fits


# In[8]:


if __name__ == '__main__':
    
    # INTIALIZATION
    
    # Define the search for maximum fitness
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Define an individual in the population 
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define characteristics of GA elements (individual, population, operators)
    toolbox = base.Toolbox()
    
    # 1 -Chromosome, DCP parameters
    toolbox.register("attr_float_omega", random.uniform, 0, 1)
    toolbox.register("attr_float_t_zero", random.uniform, 0.1, 0.95)
    
    # 2 - Individual, combination of chromosomes
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float_omega, toolbox.attr_float_t_zero), n=1)
    
    # 3 - Population, collection of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 4- Fitness Function with constraints
    toolbox.register("evaluate", fitness_fcn)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasibility, 0.0,)) # Penalty of 0.0 if constraint check fails

    # 5 - Operators: Selection, Crossover, Mutation, Population Update
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, indpb=MUTPB)
    toolbox.register("select", tools.selTournament, k=2, tournsize=5)
    toolbox.register("select_worst", tools.selWorst, k=2)
    
    # Create a population, Generation 0
    pop = toolbox.population(POPULATION)
    pop_seed = toolbox.clone(pop) # Copy of first generation 
    
    # Evaluate the intial population
    fitness_pop = set_fitness(pop)
    fitness_pop_seed = set_fitness(pop_seed)
    
    # Compute population statistics 
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    # GA PROPER
    iteration = 0
    while iteration <= GENERATION:
    
        # STATISTICS
        record = stats.compile(pop)
        print('Generation ',iteration, record)
    
        # OPERATORS
        # Selection
        parent1, parent2 = toolbox.select(pop)

        # Crossover
        if random.random() <= CXPB:
            offspring1, offspring2 = toolbox.mate(parent1, parent2)
        else:
            offspring1, offspring2 = parent1, parent2

        # Mutation
        avg_omega, sd_omega, avg_t_zero, sd_t_zero = gaussian_param(pop)
        toolbox.mutate(offspring1, [avg_omega, avg_t_zero], [sd_omega, sd_t_zero])
        mutant1 = toolbox.clone(offspring1)
        toolbox.mutate(offspring2, [avg_omega, avg_t_zero], [sd_omega, sd_t_zero])
        mutant2 = toolbox.clone(offspring2)

        # Update
        worst1, worst2 = toolbox.select_worst(pop)
        pop.remove(worst1)
        pop.remove(worst2)
        pop.append(mutant1)
        pop.append(mutant2)

        fitness_pop = set_fitness(pop)

        iteration += 1
    
    best_ind = tools.selBest(pop, 1)[0]
    best_ind_fit = best_ind.fitness.values
    print('Best Individual ', best_ind)
    print('Fitness = ', best_ind_fit)

