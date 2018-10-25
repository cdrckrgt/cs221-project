import random
import numpy as np

# pygame stuff
from ple.games.flappybird import FlappyBird
game = FlappyBird()

from ple import PLE
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

from rl.core import Env

class CustomSpace(object):
    def __init__(self, actions):
        self.actions = actions        

    def sample(self):
        return random.choice(self.actions)

    def contains(self, x):
        return x in self.actions
    
class CustomEnv(Env):
    def __init__(self, p):
        self.p = p
        self.p.reset_game()
        self.action_space = CustomSpace(self.p.getActionSet())

    def step(self, action):
        action = None if action == 0 else 119
        reward = self.p.act(action)
        obs = self.p.getScreenRGB()
        done = self.p.game_over()
        return np.array(obs), reward, done, {}

    def reset(self):
        self.p.reset_game()
        return self.p.getScreenRGB()
    
    def __del__(self):
        pass

c = CustomEnv(p)
# print (c.reset().shape)

print("actions", p.getActionSet())
import os
os._exit(0)
# keras stuff
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam

from rl.agents.sarsa import SARSAAgent

'''
class MyAgent(object):
    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        a = random.choice(self.actions)
        # if a == '119':
        for _ in range(3):
            a = random.choice(self.actions)
            if a is None: break
        print("selected action", a)
        return a

myAgent = MyAgent(p.getActionSet())
'''

nb_actions = len(p.getActionSet())

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

p.display_screen = True

from keras.callbacks import TensorBoard
tb = TensorBoard(log_dir='./logs')

sarsa.fit(c, nb_steps=1000, visualize=False, verbose=2, callbacks = [tb])

p.display_screen = True

sarsa.test(c, nb_episodes=5, visualize=False)
'''
nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
	if p.game_over(): #check if the game is over
		p.reset_game()

	obs = p.getScreenRGB()
	action = myAgent.pickAction(reward, obs)
	reward = p.act(action)
'''
