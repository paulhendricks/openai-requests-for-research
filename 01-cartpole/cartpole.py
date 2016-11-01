import gym
from playground.algos.blind import MonteCarloAgent
from playground.algos.local import HillClimbingAgent
from playground.experiment import run_experiment

# Make environment
env = gym.make('CartPole-v0')

# Run random guessing (a.k.a. monte carlo) agent
agent = MonteCarloAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=10000, watch=False)

# Run hill climbing agent
agent = HillClimbingAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=2000)
