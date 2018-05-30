############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Particle Swarm Optimization

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Particle_Swarm_Optimization, File: Python-MH-Particle Swarm Optimization.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Particle_Swarm_Optimization>

############################################################################

# Required Libraries
import pandas as pd
import numpy  as np
import random
import os

# Function: Initialize Variables
def initial_position(swarm_size = 3, min_values = [-5,-5], max_values = [5,5]):
    position = pd.DataFrame(np.zeros((swarm_size, len(min_values))))
    position['Fitness'] = 0.0
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             position.iloc[i,j] = random.uniform(min_values[j], max_values[j])
    return position

# Function: Initialize Velocity
def initial_velocity(position, min_values = [-5,-5], max_values = [5,5]):
    init_velocity = pd.DataFrame(np.zeros((position.shape[0], len(min_values))))
    for i in range(0, init_velocity.shape[0]):
        for j in range(0, init_velocity.shape[1]):
            init_velocity.iloc[i,j] = random.uniform(min_values[j], max_values[j])
    return init_velocity

# Function: Fitness Matrix
def fitness_matrix_calc(position):
    fitness_matrix = position.copy(deep = True)
    for i in range(0, fitness_matrix.shape[0]):
        fitness_matrix.iloc[i,-1] = target_function(fitness_matrix.iloc[i,0:position.shape[1]-1])
    return fitness_matrix

# Function: Individual Best
def individual_best_matrix(fitness_matrix, i_b_matrix): 
    for i in range(0, fitness_matrix.shape[0]):
        if(fitness_matrix.iloc[i,-1] < i_b_matrix.iloc[i,-1]):
            for j in range(0, fitness_matrix.shape[1]):
                i_b_matrix.iloc[i,j] = fitness_matrix.iloc[i,j]
    return i_b_matrix

# Function: Global Best
def global_best(fitness_matrix): 
    best_ind = fitness_matrix.iloc[fitness_matrix['Fitness'].idxmin(),:].copy(deep = True)
    return best_ind

# Function: Velocity
def velocity_vector(fitness_matrix, init_velocity, i_b_matrix, best_global, w = 0.5, c1 = 2, c2 = 2):
    r1 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    r2 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    velocity = pd.DataFrame(np.zeros((fitness_matrix.shape[0], init_velocity.shape[1])))
    for i in range(0, init_velocity.shape[0]):
        for j in range(0, init_velocity.shape[1]):
            velocity.iloc[i,j] = w*init_velocity.iloc[i,j] + c1*r1*(i_b_matrix.iloc[i,j] - fitness_matrix.iloc[i,j]) + c2*r2*(best_global[j] - fitness_matrix.iloc[i,j])
    return velocity

# Function: Updtade Position
def update_position(fitness_matrix, velocity, min_values = [-5,-5], max_values = [5,5]):
    updated_position = fitness_matrix.copy(deep = True)
    for i in range(0, fitness_matrix.shape[0]):
        for j in range(0, fitness_matrix.shape[1] - 1):
            if (fitness_matrix.iloc[i,j] + velocity.iloc[i,j] > max_values[j]):
                updated_position.iloc[i,j] = max_values[j]
                velocity.iloc[i,j] = 0
            elif (fitness_matrix.iloc[i,j] + velocity.iloc[i,j] < min_values[j]):
                updated_position.iloc[i,j] = min_values[j]
                velocity.iloc[i,j] = 0
            else:
                updated_position.iloc[i,j] = fitness_matrix.iloc[i,j] + velocity.iloc[i,j] 
    updated_position = fitness_matrix_calc(updated_position)
    return updated_position

# PSO Function
def particle_swarm_optimization(swarm_size = 3, min_values = [-5,-5], max_values = [5,5], iterations = 50, decay = 0, w = 0.9, c1 = 2, c2 = 2):    
    count = 0
    position = initial_position(swarm_size = swarm_size, min_values = min_values, max_values = max_values)
    init_velocity = initial_velocity(position, min_values = min_values, max_values = max_values)
    fitness_matrix = fitness_matrix_calc(position)
    i_b_matrix = fitness_matrix.copy(deep = True)
    best_global = global_best(fitness_matrix)

    while (count <= iterations):
        print("Iteration = ", count)
        
        if (count == 0):
            init_velocity = velocity_vector(fitness_matrix, init_velocity, i_b_matrix, best_global, w = w, c1 = c1, c2 = c2)
            position = update_position(fitness_matrix, init_velocity)            

        i_b_matrix  = individual_best_matrix(position, i_b_matrix)
        if (best_global[-1] > global_best(i_b_matrix)[-1]):
            best_global = global_best(i_b_matrix)   
        
        if (decay > 0):
            n = decay
            w  = w*(1 - ((count-1)**n)/(iterations**n))
            c1 = (1-c1)*(count/iterations) + c1
            c2 = (1-c2)*(count/iterations) + c2
        init_velocity = velocity_vector(position, init_velocity, i_b_matrix, best_global, w = w, c1 = c1, c2 = c2)
        position = update_position(position, init_velocity, min_values = min_values, max_values = max_values)
        count = count + 1 
        
    print(best_global)    
    return best_global

######################## Part 1 - Usage ####################################

# Function to be Minimized. Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def target_function (variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

pso = particle_swarm_optimization(swarm_size = 15, min_values = [-5,-5], max_values = [5,5], iterations = 50, decay = 2, w = 0.9, c1 = 2, c2 = 2)
