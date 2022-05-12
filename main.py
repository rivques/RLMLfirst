import rlgym

env = rlgym.make(game_speed=1)

while True:
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()

        next_obs, reward, done, gameinfo = env.step(action)

        obs = next_obs