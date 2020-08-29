# bit flipping environment for CS330

# Description: implementation of bit flipping environment
#              for testing RL
#
# Revision History
# 09/28/19    Tim Liu    started file
# 09/28/19    Tim Liu    added observation_space and action_space
#                        to BitFlipEnv
# 09/29/19    Tim Liu    changed environment to pass state and goals as
#                        numpy arrays

import tensorflow as tf
import numpy as np


class BitFlipEnv():
    '''bit flipping environment for reinforcement learning.
    The environment is a 1D vector of binary values (state vector).
    At each step, the actor can flip a single bit (0 to 1 or 1 to 0).
    The goal is to flip bits until the state vector matches the
    goal vector (also a 1D vector of binary values). At each step,
    the actor receives a goal of 0 if the state and goal vector
    do not match and a reward of 1 if the state and goal vector
    match.

    Internally the state and goal vector are a numpy array, which
    allows the vectors to be printed by the show_goal and show_state
    methods. When '''

    def __init__(self, num_bits, verbose = False):
        '''Initialize new instance of BitFlip class.
        inputs: num_bits - number of bits in the environment; must
                be an integer
                verbose - prints state and goal vector after each
                          step if True'''

        # check that num_bits is a positive integer
        if (num_bits < 0) or (type(num_bits) != type(0)):
            print("Invalid number of bits -  must be positive integer")
            return

        # number of bits in the environment
        self.num_bits = num_bits
        # randomly set the state vector
        self.state_vector = np.random.randint(0, 2, num_bits)
        # randomly set the goal vector
        self.goal_vector = np.random.randint(0, 2, num_bits)
        # whether to print debugging info
        self.verbose = verbose
        # TODO set dimensions of observation space
        self.observation_space = self.state_vector
        # TODO create action space; may use gym type
        self.action_space = num_bits
        # space of the goal vector
        self.goal_space = self.goal_vector
        # number of steps taken
        self.steps = 0

        return

    def show_goal(self):
        '''Returns the goal as a numpy array. Used for debugging.'''
        return self.goal_vector

    def show_state(self):
        '''Returns the state as a numpy array. Used for debugging.'''
        return self.state_vector

    def reset(self):
        '''resets the environment. Returns a reset state_vector
        and goal_vector as tf tensors'''

        # randomly reset both the state and the goal vectors
        self.state_vector = np.random.randint(0, 2, self.num_bits)
        self.goal_vector = np.random.randint(0, 2, self.num_bits)

        self.steps = 0

        # return as np array
        return self.state_vector, self.goal_vector


    def step(self, action):
        '''take a step and flip one of the bits.

        inputs: action - integer index of the bit to flip
        outputs: state - new state_vector (tensor)
                 reward - 0 if state != goal and 1 if state == goal
                 done - boolean value indicating if the goal has been reached'''
        self.steps += 1


        if action < 0 or action >= self.num_bits:
            # check argument is in range
            print("Invalid action! Must be integer ranging from \
                0 to num_bits-1")
            return

        # flip the bit with index action
        if self.state_vector[action] == 1:
            self.state_vector[action] = 0
        else:
            self.state_vector[action] = 1

        # initial values of reward and done - may change
        # depending on state and goal vectors
        reward = 0
        done = True

        # check if state and goal vectors are identical
        if False in (self.state_vector == self.goal_vector):
            reward = -1
            done = False

        # print additional info if verbose mode is on
        if self.verbose:
            print("Bit flipped:   ", action)
            print("Goal vector:   ", self.goal_vector)
            print("Updated state: ", self.state_vector)
            print("Reward:        ", reward)

        if done:
            #print("Solved in: ", self.steps)
            pass

        # return state as numpy arrays
        # return goal_vector in info field
        return np.copy(self.state_vector), reward, done, self.steps
