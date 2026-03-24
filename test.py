from rl_snake.env import SnakeEnv

env = SnakeEnv(height=8, width=8, seed=42)
obs = env.reset()
env.render()

done = False
while not done:
    action = env.sample_action()
    result = env.step(action)
    env.render()
    print(result.reward, result.done, result.info)
    done = result.done
