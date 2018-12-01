import random
import numpy as np

reward_values = {
    "tick" : 0.1,
    "positive" : 1.0,
    "negative" : -1.0,
    "loss" : -10.0,
    "win" : 10.0
}
# importing and creating the FlappyBird game
from ple.games.flappybird import FlappyBird
game = FlappyBird()

# putting the game in the PLE wrapper
from ple import PLE
p = PLE(game, fps=30, display_screen=True, force_fps=False, reward_values=reward_values)
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
        action = self.action_space.actions[action]
        reward = self.p.act(action)
        # TODO: find out what kind of preprocessing we can do
        obs = self.p.getScreenRGB()
        done = self.p.game_over()
        return obs, reward, done, {}

    def reset(self):
        self.p.reset_game()
        return self.p.getScreenRGB()

    def __del__(self):
        pass

    def render(self, mode):
        pass

from rl.core import Processor
from PIL import Image
import keras.backend as K

class FlappyBirdProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

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

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

# TODO: experiment with different architectures for the task
model = Sequential()
model.add(Permute((1, 2, 3), input_shape=input_shape))
model.add(Conv2D(32, (3, 3), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

processor = FlappyBirdProcessor()
memory = SequentialMemory(limit=30000, window_length=WINDOW_LENGTH)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
p.display_screen = True

from keras.callbacks import TensorBoard
from time import time
tb = TensorBoard(log_dir='../../logs/{}'.format(time()))

dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, callbacks = [tb])

p.display_screen = True

dqn.test(env, nb_episodes=5, visualize=True)
