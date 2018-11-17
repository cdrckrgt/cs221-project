import random
import numpy as np
import tensorflow as tf

random.seed(4)
tf.set_random_seed(4)

# importing and creating the FlappyBird game
from ple.games.pong import Pong
game = Pong(width = 192, height = 192,MAX_SCORE=11)

# to get nonvisual representations of the game, we need a state preprocessor
def state_preprocessor(d):
    a = []
    for k in d:
        a.append(d[k])
    return np.array(a)

# custom reward values for the game
reward_values = {
    "tick" : 0.001, # 0.1 reward for existing, incentive living longer
<<<<<<< HEAD
    "positive" : 30, # 1.0 reward for passing pipe, incentivize passing them
=======
    "positive" : 1.0, # 1.0 reward for passing pipe, incentivize passing them
>>>>>>> 3c87857d1099afd6b862d611dbe9c5ab47ea5fe2
    "negative" : -1.0,
    "loss" : -3.0, # -10.0 for dying, don't die!
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

<<<<<<< HEAD
# from rl.policy import EpsGreedyQPolicy,BoltzmannQPolicy


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
    def __init__(self, eps=1, decay = .999, mineps = .001):
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
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model,policy = EpsGreedyQPolicy(decay = .999), nb_actions=nb_actions, memory=memory, processor=processor, nb_steps_warmup=10, gamma=.99, target_model_update=1e-2)
=======
from rl.policy import EpsGreedyQPolicy,BoltzmannQPolicy

processor = None
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model,policy = None, nb_actions=nb_actions, memory=memory, processor=processor, nb_steps_warmup=10, gamma=.99, target_model_update=1e-2)
>>>>>>> 3c87857d1099afd6b862d611dbe9c5ab47ea5fe2
dqn.compile(Adam(lr=1e-4), metrics=['mae'])

p.display_screen = True

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from time import time
t = time()
tb = TensorBoard(log_dir='../../logs/pong/{}'.format(t))

filepath='../../weights/pong/best.{}.hdf5'.format(t)
cp = ModelCheckpoint(filepath, verbose=1, period=5000)
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, callbacks = [tb, cp])

p.display_screen = True

dqn.test(env, nb_episodes=5, visualize=True)
