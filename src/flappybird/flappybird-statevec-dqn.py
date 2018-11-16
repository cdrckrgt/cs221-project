import random
import numpy as np

np.random.seed(4)
random.seed(4)

import tensorflow as tf
tf.set_random_seed(4)
np.random.seed(4)

# importing and creating the FlappyBird game
from ple.games.flappybird import FlappyBird
game = FlappyBird()

# to get nonvisual representations of the game, we need a state preprocessor
def state_preprocessor(d):
    a = []
    for k in d:
        a.append(d[k])
    return np.array(a)

# custom reward values for the game
reward_values = {
    "tick" : .1, # 0.1 reward for existing, incentive living longer
    "positive" : 1.0, # 1.0 reward for passing pipe, incentivize passing them
    "negative" : -1.0,
    "loss" : -10.0, # -10.0 for dying, don't die!
    "win" : 10.0
}

# putting the game in the PLE wrapper
from ple import PLE
p = PLE(game, fps=30, display_screen=True, force_fps=True, state_preprocessor=state_preprocessor, reward_values=reward_values)
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

    def step(self, action):
        action = self.action_space.actions[action]
        reward = self.p.act(action)
        obs = self.p.getGameState()
        done = self.p.game_over()
        return obs, reward, done, {}

    def reset(self):
        self.p.reset_game()
        return self.p.getGameState()

    def __del__(self):
        pass

    def render(self, mode):
        pass

# creating the env for use with keras-rl
env = CustomEnv(p)
print(env.reset().shape)

# imports for the neural net
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam

# importing the desired drl agent
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

nb_actions = len(p.getActionSet())

input_shape = (1,) + env.reset().shape
model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy

processor = None
memory = SequentialMemory(limit=25000, window_length=1)
<<<<<<< HEAD
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, processor=processor, nb_steps_warmup=10, gamma=.99, target_model_update=0.01)
=======
policy = EpsGreedyQPolicy(0.1)
dqn = DQNAgent(model=model,policy=policy, nb_actions=nb_actions, memory=memory, processor=processor, nb_steps_warmup=100, gamma=.99, target_model_update=1e-2)
>>>>>>> 8257f48b71686ea2cb99a8d10e6aa134e05cef88
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

p.display_screen = True

from keras.callbacks import TensorBoard
from rl.callbacks import ModelIntervalCheckpoint
from time import time
t = time()
tb = TensorBoard(log_dir='../../logs/flappybird/{}'.format(t))


filepath='../../weights/flappybird/best_{}.hdf5'.format(t)
cp = ModelIntervalCheckpoint(filepath, verbose=1, period=5000)
dqn.fit(env, nb_steps=30000, visualize=False, verbose=2, callbacks = [tb, cp])

p.display_screen = True

dqn.test(env, nb_episodes=5, visualize=True)
