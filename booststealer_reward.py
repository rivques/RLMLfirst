from rlgym.utils.reward_funtions.reward_functiion import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np

class BooststealerReward(RewardFunction):
    def reset(self, initial_state: GameState):
        self.old_boost = None
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        linear_velocity = player.car_data.linear_velocity
        reward = math.vecmag(linear_velocity) / 2300 # linear scale reward from 0 to 1 for supersonic
        
        reward = 0
        
        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)