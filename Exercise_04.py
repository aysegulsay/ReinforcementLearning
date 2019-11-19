
import numpy as np
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, event):
        #  states G:1, S:2, W:3,
        self.windy = np.array([[0,0,0,3,3,1,0],
                               [0,0,0,0,3,0,3],
                               [0,3,3,0,3,0,0],
                               [2,0,3,0,0,3,0],
                               [0,0,3,0,0,0,0]])
        
        self.event = event
        self.points = {0:-1,  1:1000, 2:-1, 3:-100}
        self.y_cell = 5
        self.x_cell = 7
        self.gamma = 1
        self.epsilon = 0.1
        self.alpha = 1
        self.position = self.y_cell*self.x_cell
        self.total_move = 8 #The agent has eight actions
        self.probabilities = np.array([0.2,0.6,0.2]) 
        
        self.moves = np.array([[0, 0.25, -1, 0, 8593],#Move to the up     
                               [1, 0, -1, 1, 8599],   #Move to the upper right
                               [2, 0.5, 0, 1, 8594],  #Move to the right
                               [3, 0, 1, 1, 8600],    #Move to the lower right
                               [4, 0.25, 1, 0, 8595], #Move to the down
                               [5, 0, 1, -1, 8601],   #Move to the lower left
                               [6, 0, 0, -1, 8592],   #Move to the left
                               [7, 0, -1, -1, 8598]]) #Move to the upper left
    
   
    def get_direction(self):
        """
        Decide to take nondeterministic action according to the directions
        """
        rand = np.random.random()
        for i in range(3):
            outcome = np.sum(self.probabilities[:i+1])
            if rand < outcome:
                turn = i
                break
        return turn    
  
    def greedy_move(self, x, y):
        """
        Returns highest value of the state 
        """
        val = x*7 + y
        a_val = np.argmax(self.state_axes[val,:])
        return a_val
    
    def getStates(self,x,y):
        val = x*self.y_cell + y
        return self.value[val]
    
    def take_points(self,x, y):
        """
        Returns new reward
        """
        return self.points[self.windy[x,y]]
    
    def get_possible_moves(self):
        """
        Returns the new move or action is calculated  with probabilities
        to the given policy .
        """
        rand = np.random.random()
        for i in range(self.total_move):
            outcome = np.sum(self.moves[:i+1,1])
            if rand < outcome:
                act_val = i
                break
        turn = self.get_direction()
        if turn == 0: # shift to the right
            if act_val < 7:
                act_val += 1
            else:
                act_val = 0
        if turn == 2: # shift to the left
            if act_val > 0:
                act_val += -1
            else:
                act_val = 7
        return act_val
   
    def obtain_positions(self, x, y, act_val):
        """Move agent from specified position under the rules of the grid.
           Returns new position with values in rows and columns"""
        if act_val == 0:
            if x == 0:
                r = x
                c = y 
            else:
                r = x + self.moves[act_val,2]
                c = y + self.moves[act_val,3]
        elif act_val == 1: 
            if x == 0 and y != 6:
                r = x
                c = y + self.moves[act_val,3]
            elif x == 0 and y == 6:
                r = x
                c = y
            elif x != 0 and y == 6:
                r = x + self.moves[act_val,2]
                c = y
            else:
                r = x + self.moves[act_val,2]
                c = y + self.moves[act_val,3]
 
                # left
        elif act_val == 2: 
            if y == 6:
                r = x
                c = y 
            else:
                r = x + self.moves[act_val,2]
                c = y + self.moves[act_val,3]
                
                # lower right
        elif act_val == 3: 
            if x == 4 and y != 6:
                r = x
                c = y + self.moves[act_val,3]
            elif x == 4 and y == 6:
                r = x
                c = y
            elif x != 4 and y == 6:
                r = x + self.moves[act_val,2]
                c = y
            else:
                r = x + self.moves[act_val,2]
                c = y + self.moves[act_val,3]
                
                # down
        elif act_val == 4: 
            if x == 4:
                r = x
                c = y 
            else:
                r = x + self.moves[act_val,2]
                c = y + self.moves[act_val,3]
                
                # lower left
        elif act_val == 5: 
            if x == 4 and y != 0:
                r = x
                c = y + self.moves[act_val,3]
            elif x == 4 and y == 0:
                r = x
                c = y
            elif x != 4 and y == 0:
                r = x + self.moves[act_val,2]
                c = y
            else:
                r = x + self.moves[act_val,2]
                c = y + self.moves[act_val,3]
                
                # left
        elif act_val == 6: 
            if y == 0:
                r = x
                c = y 
            else:
                r = x + self.moves[act_val,2]
                c = y + self.moves[act_val,3]
                
                # upper left
        elif act_val == 7: 
            if x == 0 and y != 0:
                r = x
                c = y + self.moves[act_val,3]
            elif x == 0 and y == 0:
                r = x
                c = y
            elif x != 0 and y == 0:
                r = x + self.moves[act_val,2]
                c = y
            else:
                r = x + self.moves[act_val,2]
                c = y + self.moves[act_val,3]
        return int(r), int(c)
    
    

    def temporal_dif(self):
        """
            Update the value of the old state and action  by
            computing the difference between the estiomation and the sum of
            the reward and returns estimated value of next state.
        """
        #initialization
        self.value = np.zeros(self.position)
        for i in range(self.event):
            x = 3
            y = 0
            """
            Returns a boolean value whether the state is terminal
            """
            threshold = True
            while threshold:
                
                take_act = self.get_possible_moves()
                row, col = self.obtain_positions(x, y,take_act)
                reward = self.take_points(row, col)
                old_state = (x)*7 + y
                new_state = (row)*7 + col
                self.value[old_state] = self.value[old_state] + self.alpha*(reward + 
                          self.gamma*self.value[new_state] - self.value[old_state])
                x = row
                y = col
                if self.windy[x,y] in (1,2):
                    threshold = False
        return self.value
        
        
    def gridWorldQLearning(self):
        ''' Q-learning just take the action which is estimated as the best '''
        #initialization
        self.value = np.zeros(self.position)
        self.state_axes = np.zeros((self.position,self.total_move))
        self.q_learn = []
        for i in range(self.event):
            x = 3
            y = 0
            threshold = True
            while threshold:
                rand_a = np.random.random()
                if rand_a < self.epsilon:
                   take_act = self.greedy_move(x,y)
                else:
                   take_act = self.get_possible_moves()
                row, col = self.obtain_positions(x, y,take_act)
                reward = self.take_points(row, col)
                old_state = (x)*7 + y
                new_state = (row)*7 + col
                max_q = np.argmax(self.state_axes[new_state,:])
                self.state_axes[old_state,take_act] = self.state_axes[old_state,take_act] + self.alpha*(reward +
                              self.gamma*self.state_axes[new_state,max_q] - self.state_axes[old_state,take_act])
                x = row
                y = col
                if self.windy[x,y] in (1,2):
                    threshold = False
        for i in range(self.position):
            x = int(i / 7)
            y = i % 7
            max_q = np.argmax(self.state_axes[i,:])
            if self.windy[x,y] in (1,2):
                self.q_learn.append('-1')
            else:
                self.q_learn.append(chr(int(self.moves[max_q,4])))
            self.value[i] = self.state_axes[i,max_q]  
        return self.state_axes, self.value, self.q_learn
    
def show_td():
        grid = Agent(1000)
        v = grid.temporal_dif()
        v.shape = (5,7)
        print("\nExercise 4.1 State value for each visited state\n")
        print(v)
        
def drawing_q():
        grid = Agent(10000)
        q, v, q_learn = grid.gridWorldQLearning()
        q_learn = np.asarray(q_learn)
        v.shape = (5,7)
        q_learn.shape = (5,7)
        print("\nExercise 4.2 State value for each visited state\n")
        print(v)
        print("\nExercise 4.2 Resulting polic in arrow\n")
        print(q_learn)
     
        

if __name__ == "__main__":
    show_td()
    drawing_q()
 


