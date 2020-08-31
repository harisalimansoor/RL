import gym
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import argparse

from stable_baselines import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines.common.evaluation import evaluate_policy

from gym.wrappers import Monitor

import psutil
process = psutil.Process(os.getpid())

parser = argparse.ArgumentParser(description='Run TRPO algorithm on GYM Environment.')
parser.add_argument('--env', default='LunarLander-v2', type=str,
                    help='Enter name of gym environment')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--gamma', default=0.99,
                    help='The discount value.')
parser.add_argument('--batch-size', default=1024, type=int,
                    help='The number of timesteps to run per batch (horizon)')
parser.add_argument('--num-iterations', default=100, type=int,
                    help='Total number of iterations to run.')
parser.add_argument('--max-kl', default=0.01, type=float,
                    help='The Kullback-Leibler loss threshold')
parser.add_argument('--cg-iters', default=10, type=int,
                    help='The number of iterations for the conjugate gradient calculation')
parser.add_argument('--lam', default=0.98, type=float,
                    help='The GAE factor.')
parser.add_argument('--cg-damping', default=0.01, type=float,
                    help='The compute gradient dampening factor')
parser.add_argument('--vf-stepsize', default=0.0003, type=float,
                    help='The value function stepsize')
parser.add_argument('--vf-iters', default=3, type=int,
                    help='The value function’s number iterations for learning')
parser.add_argument('--save-dir', default='tmp', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()


env_name=args.env
save_dir=args.save_dir

memory_usage=[]
cpu_usage=[]
avg_returns=[]


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def wrap_env(env):
  env = Monitor(env, 'video', force=True)
  return env
  
class CustomCallback(BaseCallback):
    def __init__(self, eval_env, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.eval_env=eval_env


    def _on_step(self):
        mem_use=process.memory_info().rss/(1024*1024)
        cpu_use=process.cpu_percent()/multiprocessing.cpu_count()
        memory_usage.append(mem_use)
        cpu_usage.append(cpu_use)
        episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                               n_eval_episodes=5,
                                                               render=False,
                                                               deterministic=True,
                                                               return_episode_rewards=True)

        avg_returns.append(np.mean(episode_rewards))

        print('Memory(MB) = {} \nCPU(%)= {}'.format(mem_use,cpu_use))
        return True

class Agent():
    def __init__(self):
        self.env_name=env_name
    
    def train(self):
        eval_env=gym.make(self.env_name)
        cb=CustomCallback(eval_env)
        event_callback = EveryNTimesteps(n_steps=args.batch_size, callback=cb)

        env = gym.make(self.env_name)

        model = TRPO(MlpPolicy, env, gamma=args.gamma, timesteps_per_batch=args.batch_size, max_kl=args.max_kl, cg_iters=args.cg_iters, lam=args.lam, cg_damping=args.cg_damping, vf_stepsize=args.vf_stepsize, vf_iters=args.vf_iters, verbose=1)

        
        episode_rewards, episode_lengths = evaluate_policy(model, eval_env,
                                                               n_eval_episodes=5,
                                                               render=False,
                                                               deterministic=True,
                                                               return_episode_rewards=True)
        avg_returns.append(np.mean(episode_rewards))
        mem_use=process.memory_info().rss/(1024*1024)
        cpu_use=process.cpu_percent()/multiprocessing.cpu_count()
        memory_usage.append(mem_use)
        cpu_usage.append(cpu_use)
        
        model.learn(total_timesteps=(args.num_iterations*args.batch_size), callback=event_callback)
        
        episode_rewards, episode_lengths = evaluate_policy(model, eval_env,
                                                               n_eval_episodes=5,
                                                               render=False,
                                                               deterministic=True,
                                                               return_episode_rewards=True)
        avg_returns.append(np.mean(episode_rewards))
        mem_use=process.memory_info().rss/(1024*1024)
        cpu_use=process.cpu_percent()/multiprocessing.cpu_count()
        memory_usage.append(mem_use)
        cpu_usage.append(cpu_use)

        print('Saving model in',os.path.join(save_dir,"trpo_"+self.env_name))
        model.save(os.path.join(save_dir,"trpo_"+self.env_name))
        print('Model successfully saved')

        iterations = range(0, args.num_iterations + 1)
        plt.plot(iterations, avg_returns)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.savefig(os.path.join(save_dir,
                                '{} Average Returns.png'.format(env_name)))
        plt.clf()
        
        iterations = range(0, args.num_iterations + 1)
        plt.plot(iterations, memory_usage)
        plt.ylabel('Memory Usage (MB)')
        plt.xlabel('Iterations')
        plt.savefig(os.path.join(save_dir,
                                '{} Memory Usage.png'.format(env_name)))

        plt.clf()

        iterations = range(0, args.num_iterations + 1)
        plt.plot(iterations, cpu_usage)
        plt.ylabel('CPU Usage (%)')
        plt.xlabel('Iterations')
        plt.savefig(os.path.join(save_dir,
                                '{} CPU Usage.png'.format(env_name)))
        
        print('Plots saved in',save_dir)

    def play(self):
        print('Loading model from',os.path.join(save_dir,"trpo_"+self.env_name))
        model = TRPO.load(os.path.join(save_dir,"trpo_"+self.env_name))
        print('Model successfully loaded')

        env = wrap_env(gym.make(self.env_name))

        obs = env.reset()

        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()

if __name__ == '__main__':
    print(args)
    trpo_agent = Agent()
    if args.train:
        start=time.time()

        trpo_agent.train()

        end=time.time()
        print('\n')
        print('Maximum memory usage: {} MB'.format(np.max(np.array(memory_usage))))
        print('Average CPU usage: {} %'.format(np.mean(np.array(cpu_usage))))
        print('Time taken: {} s'.format(end-start))
        print('\n')
    else:
        trpo_agent.play()