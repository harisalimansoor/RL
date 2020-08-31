from __future__ import absolute_import, division, print_function


import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


import tensorflow as tf
import os
from gym.wrappers import Monitor
import gym
import argparse
import time
import multiprocessing

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import psutil
process = psutil.Process(os.getpid())

if not tf.executing_eagerly():
  tf.enable_eager_execution()


parser = argparse.ArgumentParser(description='Run DQN algorithm on GYM Environment.')
parser.add_argument('--env', default='LunarLander-v2', type=str,
                    help='Enter name of gym environment')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.0001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--batch-size', default=64, type=int,
                    help='Batch Size used in Replay Buffer for dataset creation.')
parser.add_argument('--num-iterations', default=40000, type=int,
                    help='Total number of iterations to run.')
parser.add_argument('--log-interval', default=200, type=int,
                    help='Interval after which a log is made.')
parser.add_argument('--eval-interval', default=1000, type=int,
                    help='Interval after which the policy is evaluated')
parser.add_argument('--save-dir', default='tmp', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()


num_iterations = args.num_iterations  

initial_collect_steps = 1000    
collect_steps_per_iteration = 1   
replay_buffer_max_length = 100000   

batch_size = args.batch_size   
learning_rate = args.lr  
log_interval = args.log_interval  

num_eval_episodes = 10   
eval_interval = args.eval_interval   

save_dir = args.save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

env_name = args.env
policy_dir = os.path.join(save_dir, 'policy_'+env_name)
memory_usage=[]
cpu_usage=[]


def wrap_env(env):
  env = Monitor(env, 'video', force=True)
  return env

class Agent():
    def __init__(self):

        self.train_py_env = suite_gym.load(env_name)
        self.eval_py_env = suite_gym.load(env_name)

        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)


        fc_layer_params = (1000,)

        self.q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=fc_layer_params)


        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=self.q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

        self.agent.initialize()

        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec, batch_size=self.train_env.batch_size, max_length=replay_buffer_max_length)

        self.random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(), self.train_env.action_spec())
        self.collect_data(self.train_env, self.random_policy, self.replay_buffer, steps=initial_collect_steps)

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

        self.iterator = iter(self.dataset)

    def compute_avg_return(self,environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]



    def collect_step(self,environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(self,env, policy, buffer, steps):
        for _ in range(steps):
            self.collect_step(env, policy, buffer)


    def train(self):
        self.agent.train = common.function(self.agent.train)

        # Reset the train step
        self.agent.train_step_counter.assign(0)

        # Evaluate the self.agent's policy once before training.
        avg_return = self.compute_avg_return(self.eval_env, self.agent.policy, num_eval_episodes)
        returns = [avg_return]
        mem_use=process.memory_info().rss/(1024*1024)
        cpu_use=process.cpu_percent()/multiprocessing.cpu_count()
        memory_usage.append(mem_use)
        cpu_usage.append(cpu_use)

        for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(collect_steps_per_iteration):
                self.collect_step(self.train_env, self.agent.collect_policy, self.replay_buffer)

            # Sample a batch of data from the buffer and update the self.agent's network.
            experience, unused_info = next(self.iterator)
            train_loss = self.agent.train(experience).loss
            step = self.agent.train_step_counter.numpy()

            if step % log_interval == 0:
                mem_use=process.memory_info().rss/(1024*1024)
                cpu_use=process.cpu_percent()/multiprocessing.cpu_count()
                memory_usage.append(mem_use)
                cpu_usage.append(cpu_use)
                print('step = {0}: loss = {1}: Memory(MB) = {2}: CPU(%)= {3}'.format(step, train_loss,mem_use,cpu_use))

            if step % eval_interval == 0:
                avg_return = self.compute_avg_return(self.eval_env, self.agent.policy, num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
        
        print('Saving Policy')
        tf_policy_saver = policy_saver.PolicySaver(self.agent.policy)
        tf_policy_saver.save(policy_dir)
        print('Policy saved in',policy_dir)


        iterations = range(0, num_iterations + 1, eval_interval)
        plt.plot(iterations, returns)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.savefig(os.path.join(save_dir,
                                '{} Average Returns.png'.format(env_name)))
        plt.clf()
        
        iterations = range(0, num_iterations + 1, log_interval)
        plt.plot(iterations, memory_usage)
        plt.ylabel('Memory Usage (MB)')
        plt.xlabel('Iterations')
        plt.savefig(os.path.join(save_dir,
                                '{} Memory Usage.png'.format(env_name)))

        plt.clf()

        iterations = range(0, num_iterations + 1, log_interval)
        plt.plot(iterations, cpu_usage)
        plt.ylabel('CPU Usage (%)')
        plt.xlabel('Iterations')
        plt.savefig(os.path.join(save_dir,
                                '{} CPU Usage.png'.format(env_name)))
        
        print('Plots saved in',save_dir)

    
    def play(self):
        print('Loading Saved Policy from',policy_dir)
        saved_policy = tf.compat.v2.saved_model.load(policy_dir)
        print('Policy Successfully Loaded')
        filename = 'GamePlay' + ".mp4"
        with imageio.get_writer(filename, fps=30) as video:
            time_step = self.eval_env.reset()
            video.append_data(self.eval_py_env.render())
            step_counter = 0
            episode_return = 0.0
            while not time_step.is_last():
                action_step = saved_policy.action(time_step)
                time_step = self.eval_env.step(action_step.action)
                video.append_data(self.eval_py_env.render())
                episode_return += time_step.reward
                print("{}. Reward: {}, action: {}".format(step_counter, episode_return, action_step.action))
                step_counter+=1


if __name__ == '__main__':
    print(args)
    dqnagent = Agent()
    if args.train:
        start=time.time()

        dqnagent.train()

        end=time.time()
        print('\n')
        print('Maximum memory usage: {} MB'.format(np.max(np.array(memory_usage))))
        print('Average CPU usage: {} %'.format(np.mean(np.array(cpu_usage))))
        print('Time taken: {} s'.format(end-start))
    else:
        dqnagent.play()