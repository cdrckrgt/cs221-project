import random
import numpy as np

# importing and creating the FlappyBird game
from ple.games.flappybird import FlappyBird
game = FlappyBird()

# putting the game in the PLE wrapper
from ple import PLE
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

# PLE wrapper doesn't follow same interface as keras-rl expects, so we
# create a custom wrapper for it
from rl.core import Env

# the space that holds the actions we can take, used in keras-rl env
class CustomSpace(object):
    '''
    A space object that defines the actions that we can take during each step.
    '''
    def __init__(self, actions):
        self.actions = actions        

    def sample(self):
        return random.choice(self.actions)

    def contains(self, x):
        return x in self.actions
    
class CustomEnv(Env):
    '''
    A custom wrapper for the Env class, allowing us to use keras-rl with games
    defined in the PLE.
    '''
    def __init__(self, p):
        self.p = p
        self.p.reset_game()
        self.action_space = CustomSpace(self.p.getActionSet())

    def step(self, action): # TODO: find out why action is only 0 or 1
        action = None if action == 0 else 119
        reward = self.p.act(action)
        # TODO: find out what kind of preprocessing we can do
        obs = self.p.getScreenRGB() 
        done = self.p.game_over()
        return np.array(obs), reward, done, {}

    def reset(self):
        self.p.reset_game()
        return self.p.getScreenRGB()
    
    def __del__(self):
        pass

# creating the env for use with keras-rl
env = CustomEnv(p)

# creating a simple random agent
"""
class RandomAgent(object):
    '''
    The simplest agent that we can create. Performs random actions at every
    step.
    '''
    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        return random.choice(self.actions)

randAgent = RandomAgent(p.getActionSet())

nb_frames = 1000
reward = 0.0

for _ in range(nb_frames):
	if p.game_over(): #check if the game is over
		p.reset_game()

	obs = p.getScreenRGB()
	action = myAgent.pickAction(reward, obs)
	reward = p.act(action)
"""

# imports for the neural net
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam

# importing the desired drl agent
from rl.agents.sarsa import SARSAAgent

nb_actions = len(p.getActionSet())

# TODO: experiment with different architectures for the task
model = Sequential()
model.add(Flatten(input_shape=(1,) + (288, 512, 3)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

sarsa = SARSAAgent(model=model, nb_actions=nb_actions)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

p.display_screen = False

from keras.callbacks import TensorBoard
from time import time
tb = TensorBoard(log_dir='./logs/{}'.format(time()))

sarsa.fit(env, nb_steps=10000, visualize=False, verbose=2, callbacks = [tb])

p.display_screen = True

sarsa.test(c, nb_episodes=5, visualize=True)
