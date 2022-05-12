from rlgym.utils.reward_funtions.reward_functiion import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np

class BooststealerReward(RewardFunction):
    def reset(self, initial_state: GameState):
        self.old_boost = 0
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        linear_velocity = player.car_data.linear_velocity
        speed_reward = math.vecmag(linear_velocity) / 2200 # linear scale reward from 0 to 1 for supersonic
        reward += 3 if speed_reward > 1 else speed_reward # add a bonus for supersonic

        net_boost_change = player.car_data.boost - self.old_boost
        reward += net_boost_change if net_boost_change > 0 else 0 # add a bonus for picking up boost

        self.old_boost = player.car_data.boost
        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)