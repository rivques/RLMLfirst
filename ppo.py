import rlgym
from stable_baselines3 import PPO
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import LiuDistanceBallToGoalReward, VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import LiuDistancePlayerToBallReward, TouchBallReward
from rlgym.utils.reward_functions.combined_reward import CombinedReward

reward_func = CombinedReward([LiuDistanceBallToGoalReward(), LiuDistancePlayerToBallReward(), VelocityBallToGoalReward(), TouchBallReward()], [2, 1, .5, 2])
#reward_func = LiuDistanceBallToGoalReward()
#Make the default rlgym environment
env = rlgym.make(
    reward_fn=reward_func,
)

#Initialize PPO from stable_baselines3
model = PPO("MlpPolicy", env=env, verbose=1)

#Train our agent!
model.learn(total_timesteps=int(1e6))