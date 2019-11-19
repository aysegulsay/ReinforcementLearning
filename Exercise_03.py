
from itertools import product

import numpy as np
import matplotlib.pyplot as plt



class Agent:
     #Define each direction as up ,down ,right, left and diagonal moves
    north = np.array([-1, 0])
    south = np.array([1, 0])
    west = np.array([0, -1])
    east = np.array([0, 1])
    south_east = np.array([1, 1])
    south_west = np.array([1,-1])
    north_west = np.array([-1,-1])
    north_east = np.array([-1, 1])
to_the_east = np.array([[1, -1], [1, 1]])
to_the_west = to_the_east.T
class grid_world(object):
  
    
    def __init__(self):
        #building array for grid world
        self.array = np.array([['*', '*', '*', ' ', '*', '*', ' ', '*', '*'],
                               [' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', '*'],
                               [' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', ' '],
                               [' ', ' ', ' ', '*', '*', '*', ' ', 'X', ' '],
                               [' ', ' ', 'X', 'X', 'X', 'X', 'X', ' ', ' '],
                               [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', ' '],
                               [' ', 'X', 'X', 'X', 'X', 'X', 'X', 'G', ' '],
                               [' ', '*', '*', '*', 'X', '*', '*', ' ', 'X'],
                               [' ', '*', '*', '*', ' ', '*', '*', ' ', 'X']])
        self.last_position = False
        self.directions = [Agent.north, Agent.east, Agent.south, Agent.west  ]
        self.diagonal = [Agent.north, Agent.east, Agent.south, Agent.west,
                         Agent.south_west,Agent.south_east,
                         Agent.north_east, Agent.north_west ]
        
    """ Calculate each next move with current state and action values. """
    def next_direction(self, position, next_move):
        if self.array[position[0], position[1]] == 'G':
            return position, 0.
        cell_update = position + next_move
        if (not cell_update[0] in range(0, 9) or not cell_update[1] in range(0, 9)):
            return position, -5. #attempt to live the grid
        if self.array[tuple(cell_update)] == ' ':
            position = cell_update
            return position, -1.
        if self.array[tuple(cell_update)] == 'X':
            return position, -20.#cell marked X, it will receive -20 points .
        if self.array[tuple(cell_update)] == '*':
            position = cell_update
            return position, 5. #a cell marked *, it will receive 5 points
        if self.array[tuple(cell_update)] == 'G':
            position = cell_update
            self.last_position = True
            return position, 100.#When it reaches cell G, it will receive 100 points and

""" Compute state and action and update them with using probabilities"""
class grid_env(grid_world):
    def next_direction(self, position, next_move):
        if self.array[tuple(position)] == 2:
            return [[1], [position], [0.]]
        next_cell_update = [np.dot(to_the_east, next_move), next_move, np.dot(to_the_west, next_move)]
        transition_prob = [0.15, 0.7, 0.15]
        position_prob = []
        points = []
        for new_position in next_cell_update:
            value, Q = super(grid_env, self).next_direction(position, new_position)
            position_prob.append(value)
            points.append(Q)
            return transition_prob, position_prob, points


def build_grid(grid, grph):  
    fig = plt.figure()
    fig_ax = fig.add_subplot(1,1,1)
    fig_ax.set_title(grph)
    fig_ax.set_xticks(np.arange(0.5,10.5,1))
    fig_ax.set_yticks(np.arange(0.5,9.5,1))
    fig_ax.grid(color='w', linestyle='-', linewidth=1)
    fig_ax.imshow(grid, interpolation='nearest', cmap='GnBu')
    return fig_ax


def policy_eval():
    """ Create policy from 3.1 and do iterative policy evaluation. """
   
    action_prob = [0.125, 0.625, 0.125, 0.125]# actions with probabilities
    data = grid_world()
    state_axis = np.zeros((9, 9))#initialize states
    threshold = .1
    prior_state = np.ones((9, 9))
   
    while np.abs(state_axis - prior_state).max() > threshold:
        for x, y in product(range(9), repeat=2):
            prior_state = state_axis.copy()
            if data.array[x, y] == 'X':
                continue
            updated_values = [data.next_direction(np.array([x, y]), next_move)
        for next_move in data.directions]#Updating states with directions
            Sum_Expectation = np.dot(action_prob,
                       [points_val + 0.9 * state_axis[position[0], position[1]]
                        for position, points_val in updated_values])
            state_axis[x, y] = Sum_Expectation
    print("\nExercise 3.1 Shows Value functions for the policy\n")
    print(state_axis)
    build_grid(state_axis, "Shows Value functions for the policy")

""" Exercise 3.2 """
""" solve value iteration with deterministic policy """
""" find the best action and update the state according to the cells """

def draw_graph(state_axis, grph, data):
    
    directions = []
    for x, y in product(range(9), repeat=2):
        if data.array[x, y] in ['X', 'G']:
            continue
       
        updated_values = [data.next_direction(np.array([x, y]), next_move)
                       for next_move in data.directions]
        best_value = [points_val + 0.9 * state_axis[position[0], position[1]]
                for position, points_val in updated_values]
        delta = max(best_value)
        max_a = np.where(np.array(best_value) == delta)
        directions.extend([[x, y, move[0], move[1]] for move in
                           np.array(data.directions)[max_a]])

    build_grid(state_axis, "#3.2 Policy Iteration for optimal policy π*(s)")
    directions = np.array(directions)
    plt.quiver(directions[:, 1], directions[:, 0], directions[
               :, 3], -directions[:, 2], pivot='tail', scale=20)
    plt.ylim(9, -1)
    plt.xlim(-1, 9)
    
#Deterministic policy with value iteration
def deterministic_policy():
     #initialize values
    data = grid_world()
    state_axis = np.zeros((9, 9))
    threshold = .1
    prior_state = np.ones((9, 9))
    while np.abs(prior_state - state_axis).max() > threshold:
        prior_state = state_axis.copy()
        for x, y in product(range(9), repeat=2):
            if data.array[x, y] == 'X':
                continue
            updated_values = [data.next_direction(np.array([x, y]), next_move)
                           for next_move in data.directions]
            Sum_Expectation= max([points_val + 0.9 * state_axis[position[0], position[1]]
                     for position, points_val in updated_values])
            state_axis[x, y] = Sum_Expectation

    print("\n Exercise 3.2 The optimal value V*(s) for each cell \n")
    print(state_axis)
    draw_graph(state_axis, "\n The optimal value V*(s) for each cell.", data)


""" Exercise 3.3 """
""" with using diagonal moves -calculate value iteration with deterministic policy """
""" find the best action and update the state according eight neighbouring cells """
def draw_diagonal(state_axis, grph, data):
    
    diagonal = []
    for x, y in product(range(9), repeat=2):
        if data.array[x, y] in ['X', 'G']:
            continue
       
        updated_values = [data.next_direction(np.array([x, y]), next_move)
                       for next_move in data.diagonal]
        best_value = [points_val + 0.9 * state_axis[position[0], position[1]]
                for position, points_val in updated_values]
        delta = max(best_value)
        max_a = np.where(np.array(best_value) == delta)
        diagonal.extend([[x, y, move[0], move[1]] for move in
                           np.array(data.diagonal)[max_a]])

    build_grid(state_axis, "#3.3 Extended moves-optimal policy π*(s)")
    diagonal = np.array(diagonal)
    plt.quiver(diagonal[:, 1], diagonal[:, 0], diagonal[
               :, 3], -diagonal[:, 2], pivot='tail', scale=20)
    plt.ylim(9, -1)
    plt.xlim(-1, 9)
    
def diagonal_policy():
     #initialize values
    data = grid_world()
    state_axis = np.zeros((9, 9))
    threshold = .1
    prior_state = np.ones((9, 9))
    while np.abs(prior_state - state_axis).max() > threshold:
        prior_state = state_axis.copy()
        for x, y in product(range(9), repeat=2):
            if data.array[x, y] == 'X':
                continue
            updated_values = [data.next_direction(np.array([x, y]), next_move)
                           for next_move in data.diagonal]
            Sum_Expectation= max([points_val + 0.9 * state_axis[position[0], position[1]]
                     for position, points_val in updated_values])
            state_axis[x, y] = Sum_Expectation

    print("\nExercise 3.3 The Value Iteration algorithm to compute the optimal value V*(s)\n")
    print(state_axis)
    draw_diagonal(state_axis, "with diagonal moves", data)


#Exercise 3.4
""" using non-deterministic poliy compute each state and action with using
    transition probabilities """
def draw_prob_model(state_axis, grph, data, get_reward):
   
    directions = []
    for x, y in product(range(9), repeat=2):
        if data.array[x, y] in ['X', 'G']:
            continue
        delta = max(get_reward[x, y, :])
        max_a = np.where(np.array(
            get_reward[x, y, :]) == delta)
        # make list of best actions to draw as arrow diagram
        directions.extend([[x, y, move[0], move[1]] for move in
                           np.array(data.directions)[max_a]])
    build_grid(state_axis, "#3.4 Actions V*(s) and  π*(s) with arrows")
    directions = np.array(directions)
    plt.quiver(directions[:, 1], directions[:, 0], directions[
               :, 3], -directions[:, 2], pivot='tail', scale=20)
    plt.ylim(9, -1)
    plt.xlim(-1, 9)


def prob_model():
    data = grid_env()
    state_axis = np.zeros((9, 9))
    get_reward = np.zeros((9, 9, 4))
    threshold = .1
    prior_state = np.ones((9, 9))
    while np.abs(prior_state - state_axis).max() > threshold:
        prior_state = state_axis.copy()
        for x, y in product(range(9), repeat=2):
            if data.array[x, y] == 'X':
                continue 
            updated_values = [data.next_direction(np.array([x, y]), next_move)
                           for next_move in data.directions]
            prob_value, positions, total_points = zip(*updated_values)
            sum_v = [[state_axis[value[0], value[1]] for value in position_prob]
                          for position_prob in positions]
            Total_outcome= [(probs * (points + 0.9 * np.array(values))).sum()
                 for probs, values, points
                 in zip(prob_value, sum_v, total_points)]
            get_reward[x, y, :] = Total_outcome
            state_axis[x, y] = max(Total_outcome)
    print("\nExercise 3.4 with probablities of  return\n")
    print(state_axis)
    #build_grid(state_axis, "# 3.4 Value iteration")
    draw_prob_model(state_axis, "with Transition probabilities",data, get_reward)



    

if __name__ == "__main__":
    
    policy_eval()
    deterministic_policy()
    diagonal_policy()
    prob_model()
    plt.show()
