# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:09:31 2019

@author: sayze
"""
# Exercise 2.3 Answer
# The student considers two policies for choosing the tasks: Policy A anb Policy B
# We defined the student_task_values according to the given in the exercise 1.1 in the assignment sheet
student_task_values = [(1, 12, 0.25), 
                       (2, 4, 0.4),
                       (3, 10, 0.35),
                       (4, 5, 0.6),
                       (5, 7, 0.45),
                       (6, 3, 0.5),
                       (7, 50, 0.15)] # They are the values in order Task number, Points, Probabilities


def policyReturn(abc, n): #method definition for expected return for both policies
    #abc for policy a, b or c and N variable student can attempt only 𝑁=10 
    if len(abc) > 0 and n < 10: # Three policies with attemps 10 times to solve a task
        student_task = abc[0] #define a one task to one policy
        n += 1 # raise the number of attemps by adding 1
        expectedreturn = (student_task[2] * 
                         (student_task[1] + policyReturn(abc[1:], n))
                         + (1-student_task[2]) 
                         * (policyReturn(abc, n)))
        #expected return calculated with adding a task to the policy
        # and the subtracting task from the policy and calling the method again
        return expectedreturn  # policy succeded
    else:
        return 0     # policy fail
    
#Policy definitions: 
#Policy A : work on the tasks in sequential order
#We sorted according to the sequential order given in the exercise 1.1
policyA = sorted(student_task_values, key=lambda student_task:student_task[0])
print('Policy A Tasks: ' + '\n' + str(policyA))
policyA_return = policyReturn(policyA, 0)
print('\nPolicy A Expected return = ' + str(policyA_return))

#Policy B : work on the tasks in the order of increasing difficulty
#We sorted tasks according the highest reward and lowest probility to lowest reward and highest probility
policyB = sorted(student_task_values, key=lambda student_task:student_task[2])
print('\nPolicy B Tasks: ' + '\n' + str(policyB))
policyB_return = policyReturn(policyB, 0)
print('\nPolicy B Expected return = ' + str(policyB_return))


#Ex 2.4 answer
#improved policy C that has a higher expected return than both of the above policies
#B taking first three tasks with low probility and then other task are taken from highest to lowest reward
policyC = sorted(student_task_values, key=lambda student_task:student_task[1], reverse = True)
print('\nPolicy C Tasks: ' + '\n' + str(policyC))

policyC_return = policyReturn(policyC, 0)
print('\nPolicy A Expected return = ' + str(policyC_return))

