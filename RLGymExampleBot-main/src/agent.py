import os
from stable_baselines3 import PPO

class Agent:
    def __init__(self):
        # If you need to load your model from a file this is the time to do it
        # You can do something like:
        #
        # self.actor = # your Model
        #
        # cur_dir = os.path.dirname(os.path.realpath(__file__))
        # with open(os.path.join(cur_dir, 'model.p'), 'rb') as file:
        #     model = pickle.load(file)
        # self.actor.load_state_dict(model)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.actor = PPO.load(cur_dir + "\\ppo_boost_bot")

    def act(self, state):
        # Evaluate your model here
        action = self.actor.predict(state)
        return action
