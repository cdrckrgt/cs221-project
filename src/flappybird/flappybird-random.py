import random
import numpy as np

# importing and creating the FlappyBird game
from ple.games.flappybird import FlappyBird
game = FlappyBird()

# putting the game in the PLE wrapper
from ple import PLE
p = PLE(game, fps=30, display_screen=True, force_fps=False, state_preprocessor=state_preprocessor, reward_values=reward_values)
p.init()

# creating a simple random agent
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
