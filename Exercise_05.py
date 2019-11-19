


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, pyplot as plt
import itertools as it



class game_BeCareful(object):
    def __init__(self):
      
        self.action = ['hit', 'stick']    # game action constants 
        self.N0 = 10   # constant
        self.gamma = 1 # gamma discounting rate
        self.episodes = 1100  # number of whole episodes
        self.reward = [1, 0, -1]  # outcome: win reward +1, lose reward -1, or draw reward 0
        self.color = np.array([[0, 0.3],[1, 0.7]]) # red is 0 and probability .3 or black is probability .7
        self.dealer_s = [[3,6],[6,9],[9,12]] # cuboid states for dealer
        self.player_s = [[1,6],[4,9],[7,12],[10,15],[13,18],[16,21]] # cuboid  states for player
        
    
    def draw_red_or_black_card(self):
         # Each draw from the deck results in a value between 3 and 12
        value = np.random.randint(3,13)
        card_color = np.random.random()
        if card_color < self.color[0,1]:
            color_rb = 0
        else:
            color_rb = 1
        next_card = [value,color_rb]
        return next_card

    def check_card_sum(self, card_sum):      
        # checking all cards sum
        ace = False
        total = 0
        for i in range(len(card_sum)):
            next_card = card_sum[i]
            if next_card[1] == 0:
                total -= next_card[0]
            else:
                total += next_card[0]
        return total
    
  
    def draw_player(self):
        # player draw red or black card  
        card_sum = []
        card_sum.append(self.draw_red_or_black_card())
        return self.check_card_sum(card_sum)
    
    
    def dealer_firstcard(self, next_card, player_sum):
        # The dealer always sticks on any sum of 15 or greater, and hits otherwise.
        card_sum = []
        card_sum.append(next_card)
       
        while self.check_card_sum(card_sum) > 0 and self.check_card_sum(card_sum) < 15:
            card_sum.append(self.draw_red_or_black_card())
        return self.check_card_sum(card_sum)
 
    
   
    def draw_black_card(self):
        # Start game with black card
        dealer_first_card = self.draw_red_or_black_card()[0]
        player_sum = np.abs(self.draw_player())
        return [dealer_first_card, player_sum]
    
   
    def advance(self, state, action):
        # function (s′, r) = advance(s, a), which takes as input a state  and an action 
        # returns the next game state and reward, given current state and action.
        terminal_state = False
        # player turn
        if action == 'hit':
            next_card = self.draw_red_or_black_card()
            if next_card[0] == 0:
                state[1] += next_card[0]
            else:
                state[1] -= next_card[0]   
            if state[1] < 1 or state[1] > 21:
                # if player is bust terminate game
                terminal_state = True
                reward = -1
            else: 
                reward = 0
     
        # dealer turn
        else: # action=stick
            terminal_state = True
            # dealer goes bust
            dealer_sum = self.dealer_firstcard([state[0],1],state[1])
            if dealer_sum < 0 or dealer_sum > 21:
                reward = 1
            # find a winner
            else:
                if dealer_sum > state[1]:
                    reward = -1
                elif dealer_sum == state[1]:
                    reward = 0
                else:
                    reward = 1
        return terminal_state,state,reward
    
 
    def choose_action_greedy(self, state):
        #  function which returns greedy action.
        actions_greedy = np.argmax(self.Q_value[state[0]-1,state[1]-1,:])
        return actions_greedy
    
    
    def run_sarsa_lambda_control(self):
        # sarsa lambda algorithm

        lambd = 0
        lambd_r = 0
        all_rewards = np.zeros(11) # for 11 lambda values
    
        while lambd <= 1:
             # run algorithm with lambda values 0, 0.1, 0.2, ..., 1 
            
            self.Q_value = np.zeros([12,21,2])  # Q_value for value state 
            self.Ns_value = np.zeros([12,21])   # Ns_value is the number of times state s has been visited
            self.Nsa_value = np.zeros([12,21,2])  # Nsa_value is the number of times action a has been selected from state s.
            self.eligibility_trace = np.zeros([12,21,2])
            accumulated_reward = 0
            for i in range(1100):
                state = self.draw_black_card()
                action_time = np.random.randint(0,2)
                self.Ns_value[state[0]-1, state[1]-1] += 1
                self.Nsa_value[state[0]-1, state[1]-1, action_time] += 1
                other_state = True
                while other_state:
                    # calculate accumulated reward
                    terminal_state,number_states,reward = self.advance(state,self.action[action_time])
                    if i >= 1000:
                        accumulated_reward += reward
                    if number_states[1] < 1 or number_states[1] > 21:
                        break
                    if terminal_state:
                        other_state = False
                    epsilon_value = self.N0 / (self.N0 + self.Ns_value[number_states[0]-1,number_states[1]-1])
                    if i < 1000:   # first 1000 episodes
                        first_action = np.random.random()
                        if first_action < epsilon_value: # check for less than epsilon
                            number_action = np.random.randint(0,2)
                        else:
                            number_action = self.choose_action_greedy(number_states) 
                        self.Ns_value[number_states[0]-1, number_states[1]-1] += 1    # how many number of times state visited is updated
                        self.Nsa_value[number_states[0]-1, number_states[1]-1, number_action] += 1
                        
                        # delta is calculated
                        calculate_delta = reward + self.Q_value[number_states[0]-1, number_states[1]-1, number_action] - self.Q_value[state[0]-1, state[1]-1, action_time]
                        self.eligibility_trace[state[0]-1, state[1]-1, action_time] += 1
                        alpha_value = 1/self.Nsa_value[number_states[0]-1, number_states[1]-1, number_action]  
                        self.Q_value += alpha_value*calculate_delta*self.eligibility_trace # q_value updated
                        self.eligibility_trace = self.gamma*lambd*self.eligibility_trace

                    else:
                        # state and action updated
                        number_action = self.choose_action_greedy(number_states)
                    state = number_states
                    action_time = number_action
                    
            all_rewards[lambd_r] = accumulated_reward
            lambd += 0.1  # lambda is incremented by 0.1
            lambd_r += 1
        lambd = np.linspace(0, 1, num=11)
        return lambd,all_rewards 
    
    
    def get_state_features(self, state, action):
        # Transforms a given state into a feature representation based on the given state
        features = [] # show which features correspond to which the given state 
        for i in range(len(self.player_s)):
            for j in range(len(self.dealer_s)):
                player_range = self.player_s[i]
                dealer_range = self.dealer_s[j]
                if player_range[0] <= state[1] <= player_range[1] and dealer_range[0] <= state[0] <= dealer_range[1]:
                    features.append(1)
                else:
                    features.append(0)
        # Gives a 36 feature vector, where the order depends if the action is hit or stick            
        if action == 'hit':
            return np.concatenate([np.array(features), np.zeros(18)])
        else:
            return np.concatenate([np.zeros(18),np.array(features)])
        
    
    

    
    def coarse_coding_function_approximation(self):
        # Value function approximation using coarse coding with linear gradient descent method
        lambd = 0
        lambd_r = 0
        all_rewards = np.zeros(11) # reward for every value of lambda
        epsilon_value = 0.1 # use a constant exploration value 0.1
        alpha_value = 0.05 # use constant step-size with value 0.05
        while lambd <= 1: # continue until lambda=1
            self.theta = np.random.random(36)*0.5 # theta : parameters of the linear function approximator
            accumulated_reward = 0
            
            for i in range(1100):
                state = self.draw_black_card()
                action_time = np.random.randint(0,2)  # choose action between 0 and 1 
                self.eligibility_trace = np.zeros(36)  # intialize value of eligibility trace
                current_action = self.get_state_features(state,self.action[action_time])
                other_state = True
                while other_state:
                    self.eligibility_trace[current_action==1] += 1  
                    terminal_state,number_states,reward = self.advance(state,self.action[action_time])
                    if i >= 1000: # reward for episodes
                        accumulated_reward += reward
                    if number_states[1] > 21:
                        break
                    if terminal_state:
                        other_state = False
                    if i < 1000:
                        calculate_delta = reward - current_action*self.theta.T  # calculate delta according to rewards      
                        first_action = np.random.random()
                        if first_action < 1-epsilon_value:  
                            Q_action_value = float(-100000)
                            for act in range(len(self.action)):
                                current_features = self.get_state_features(number_states,self.action[act])
                                Q_value =  sum(self.theta*current_features)   # calculate current Q values
                                if Q_value > Q_action_value:
                                    Q_action_value = Q_value
                                    number_action = act
                                    current_action = current_features
                                    
                        else:
                            number_action = np.random.randint(0,2)
                            current_action = self.get_state_features(number_states,number_action)
                            Q_action_value = sum(self.theta*current_action) 

                        calculate_delta += self.gamma*Q_action_value   
                        self.theta += alpha_value*calculate_delta*self.eligibility_trace # theta is updated
                        self.eligibility_trace = self.gamma*lambd*self.eligibility_trace # delta is updated
                    else:
                        Q_action_value = float(-100000)
                        # calculate Q value function
                        for act in range(len(self.action)):
                            current_features = self.get_state_features(number_states,self.action[act])
                            Q_value =  sum(self.theta*current_features)
                            if Q_value > Q_action_value:
                                Q_action_value = Q_value
                                number_action = act
                                current_action = current_features
                    state = number_states
                    action_time = number_action
                    
            all_rewards[lambd_r] = accumulated_reward
            lambd += 0.1
            lambd_r += 1
        lambd = np.linspace(0, 1, num=11)
        return lambd,all_rewards
    
def sarsa_lamda_draw():

    test_play = game_BeCareful()
    action = test_play.advance([1,1],1)
    
    print("\nExercise 5.1 Play out the dealer’s cards and return the final reward and terminal state.\n")
    print(action)
    lambd,reward = test_play.run_sarsa_lambda_control()
    print("\nExercise 5.2 Sarsa Lambda - Plotting of the Accumulated Reward\n")
    print(reward)
    plt.plot(lambd,reward,'o-')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'Reward')
    plt.title("Accumulated Reward vs lambda ")
    plt.show()


def func_approximation_draw():  
    test_play = game_BeCareful()
    lambd,reward = test_play.coarse_coding_function_approximation()
    print("\nExercise 5.3 Value Function Approximator - Plotting the of Accumulated Reward \n")
    print(reward)
    plt.plot(lambd,reward,'o-')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'Reward')
    plt.title("Accumulated Reward vs lambda ")
    plt.show()
    
if __name__ == "__main__":
    sarsa_lamda_draw()
    func_approximation_draw()
       
