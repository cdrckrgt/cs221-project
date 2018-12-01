import random
import numpy as np

np.random.seed(4)
random.seed(4)

import tensorflow as tf
tf.set_random_seed(4)

# importing and creating the pixelcopter game
from ple.games.pixelcopter import Pixelcopter
game = Pixelcopter(width = 192, height = 192)

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

from rl.util import *

class Policy(object):
    """Abstract base class for all implemented policies.
    Each policy helps with selection of action to take on an environment.
    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods:
    - `select_action`
    # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        """Return configuration of the policy
        # Returns
            Configuration as dict
        """
        return {}


class EpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy
    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.3, decay = .999, mineps = .1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps
        self.decay = decay
        self.mineps = mineps

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions-1)
        else:
            action = np.argmax(q_values)
        if self.eps > self.mineps:
            self.eps *= self.decay
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy
        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

processor = None
memory = SequentialMemory(limit=100000, window_length=1)
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.2, value_min = .05, value_test = .02,
                            nb_steps = 75000)
# policy = None
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy = policy, memory=memory, processor=processor, nb_steps_warmup=100, gamma=.99, target_model_update=1e-2)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

p.display_screen = True

from keras.callbacks import TensorBoard
from rl.callbacks import ModelIntervalCheckpoint
from time import time
t = time()
tb = TensorBoard(log_dir='../../logs/pixelcopter/{}'.format(t))
# filepath='../../weights/pixelcopter/best_{}.hdf5'.format(t)
# cp = ModelIntervalCheckpoint(filepath, verbose=1, interval=5000)
dqn.fit(env, nb_steps=150000, visualize=False, verbose=2, callbacks = [tb])

p.display_screen = True

dqn.test(env, nb_episodes=15, visualize=True)
