import rlgym
from stable_baselines3 import PPO

env = rlgym.make(game_speed=1, self_play=True)

actor = PPO.load("ppo_boost_bot", env)

while True:
    obs = env.reset()
    done = False

    while not done:
        action = actor.predict(obs)

        next_obs, reward, done, gameinfo = env.step(action)

        obs = next_obs