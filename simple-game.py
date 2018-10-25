import random

from ple.games.flappybird import FlappyBird

game = FlappyBird()

from ple import PLE

p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

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

nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
	if p.game_over(): #check if the game is over
		p.reset_game()

	obs = p.getScreenRGB()
	action = myAgent.pickAction(reward, obs)
	reward = p.act(action)
