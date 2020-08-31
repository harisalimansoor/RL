import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt
import time


import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from gym.wrappers import Monitor

import psutil
process = psutil.Process(os.getpid())

if not tf.executing_eagerly():
  tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Run A3C algorithm on GYM Environment.')
parser.add_argument('--env', default='LunarLander-v2', type=str,
                    help='Enter name of gym environment')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=1000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='tmp', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()

memory_usage=[]
cpu_usage=[]

class ACModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ACModel, self).__init__()
    self.state_shape = state_size
    self.action_shape = action_size
    self.dense1 = layers.Dense(100, activation='relu')
    self.policy_logits = layers.Dense(action_size)
    self.dense2 = layers.Dense(100, activation='relu')
    self.values = layers.Dense(1)

  def call(self, inputs):
    # Forward pass
    x = self.dense1(inputs)
    logits = self.policy_logits(x)
    v1 = self.dense2(inputs)
    values = self.values(v1)
    return logits, values

def print_info(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps,
           cpu_use,
           mem_use):

  if global_ep_reward == 0:
    global_ep_reward = episode_reward
  else:
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
  print(
      f"Episode: {episode} | "
      f"Moving Average Reward: {int(global_ep_reward)} | "
      f"Episode Reward: {int(episode_reward)} | "
      f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
      f"Steps: {num_steps} | "
      f"Worker: {worker_idx} | "
      f"Memory(MB): {mem_use} | "
      f"CPU(%): {cpu_use} | "
  )
  result_queue.put(global_ep_reward)
  return global_ep_reward

def wrap_env(env):
  env = Monitor(env, 'video', force=True)
  return env



class Agent():
  def __init__(self):
    self.env_name = args.env
    save_dir = args.save_dir
    self.save_dir = save_dir
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    env = gym.make(self.env_name)
    self.state_shape = env.observation_space.shape[0]
    self.action_shape = env.action_space.n
    self.opt = tf.compat.v1.train.AdamOptimizer(args.lr, use_locking=True)
    print(self.state_shape, self.action_shape)

    self.global_model = ACModel(self.state_shape, self.action_shape)  # global network
    self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_shape)), dtype=tf.float32))

  def train(self):
    start=time.time()

    res_queue = Queue()

    workers = [Worker(self.state_shape,
                      self.action_shape,
                      self.global_model,
                      self.opt, res_queue,
                      i, game_name=self.env_name,
                      save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

    for i, worker in enumerate(workers):
      print("Starting worker {}".format(i))
      worker.start()

    moving_avg_rewards = []  # print_info episode reward to plot
    while True:
      reward = res_queue.get()
      if reward is not None:
        moving_avg_rewards.append(reward)
      else:
        break
    [w.join() for w in workers]

    plt.plot(moving_avg_rewards)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.savefig(os.path.join(self.save_dir,
                             '{} Moving Average.png'.format(self.env_name)))
    plt.clf()

    plt.plot(memory_usage)
    plt.ylabel('Memory Usage (MB)')
    plt.xlabel('Step')
    plt.savefig(os.path.join(self.save_dir,
                             '{} Memory Usage.png'.format(self.env_name)))

    plt.clf()

    plt.plot(cpu_usage)
    plt.ylabel('CPU Usage (%)')
    plt.xlabel('Step')
    plt.savefig(os.path.join(self.save_dir,
                             '{} CPU Usage.png'.format(self.env_name)))

    end=time.time()
    print('\n')
    print('Maximum memory usage: {} MB'.format(np.max(np.array(memory_usage))))
    print('Average CPU usage: {} %'.format(np.mean(np.array(cpu_usage))))
    print('Time taken: {} s'.format(end-start))

    exit()



  def play(self):
    env = wrap_env(gym.make(self.env_name))
    state = env.reset()
    env._max_episode_steps = 1000
    model = self.global_model
    model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.env_name))
    print('Loading model from: {}'.format(model_path))
    model.load_weights(model_path)
    done = False
    step_counter = 0
    reward_sum = 0
    
    try:
      while not done:
        env.render(mode='rgb_array')
        policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        action = np.argmax(policy)
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
        step_counter += 1
    except KeyboardInterrupt:
      print("Received Keyboard Interrupt. Shutting down.")
    finally:
      env.close()

    


class Episode_Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []


class Worker(threading.Thread):
  # Set up global variables across different threads
  global_episode = 0
  # Moving average reward
  global_moving_average_reward = 0
  best_score = 0
  save_lock = threading.Lock()

  def __init__(self,
               state_size,
               action_size,
               global_model,
               opt,
               result_queue,
               idx,
               game_name='LunarLander-v2',
               save_dir='tmp'):
    super(Worker, self).__init__()
    self.state_shape = state_size
    self.action_shape = action_size
    self.result_queue = result_queue
    self.global_model = global_model
    self.opt = opt
    self.local_model = ACModel(self.state_shape, self.action_shape)
    self.worker_idx = idx
    self.env_name = game_name
    self.env = gym.make(self.env_name)
    self.env._max_episode_steps = 1000
    self.save_dir = save_dir
    self.eps_loss = 0.0

  def run(self):
    total_step = 1
    mem = Episode_Memory()
    while Worker.global_episode < args.max_eps:
      current_state = self.env.reset()
      mem.clear()
      eps_reward = 0.
      eps_steps = 0
      self.eps_loss = 0

      time_count = 0
      done = False
      while not done:
        logits, _ = self.local_model(
            tf.convert_to_tensor(current_state[None, :],
                                 dtype=tf.float32))
        probs = tf.nn.softmax(logits)

        action = np.random.choice(self.action_shape, p=probs.numpy()[0])
        new_state, reward, done, _ = self.env.step(action)
        if done:
          reward = -1
        eps_reward += reward
        mem.store(current_state, action, reward)

        if time_count == args.update_freq or done:
          # Calculate gradient wrt to local model. We do so by tracking the
          # variables involved in computing the loss by using tf.GradientTape
          with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done,
                                           new_state,
                                           mem,
                                           args.gamma)
          self.eps_loss += total_loss
          # Calculate local gradients
          grads = tape.gradient(total_loss, self.local_model.trainable_weights)
          # Push local gradients to global model
          self.opt.apply_gradients(zip(grads,
                                       self.global_model.trainable_weights))
          # Update local model with new weights
          self.local_model.set_weights(self.global_model.get_weights())

          mem.clear()
          time_count = 0

          if done:  # done and print information
            cpu_use = process.cpu_percent()/multiprocessing.cpu_count()
            mem_use = process.memory_info().rss/(1024*1024)
            with Worker.save_lock:
              Worker.global_moving_average_reward = \
                print_info(Worker.global_episode, eps_reward, self.worker_idx,
                      Worker.global_moving_average_reward, self.result_queue,
                      self.eps_loss, eps_steps, cpu_use, mem_use)
              Worker.global_episode += 1
              memory_usage.append(mem_use)
              cpu_usage.append(cpu_use)
            # We must use a lock to save our model and to print to prevent data races.
            if eps_reward > Worker.best_score:
              with Worker.save_lock:
                print("Saving best model to {}, "
                      "episode score: {}".format(self.save_dir, eps_reward))
                self.global_model.save_weights(
                    os.path.join(self.save_dir,
                                 'model_{}.h5'.format(self.env_name))
                )
                Worker.best_score = eps_reward
            # Worker.global_episode += 1
        eps_steps += 1

        time_count += 1
        current_state = new_state
        total_step += 1
    self.result_queue.put(None)

  def compute_loss(self,
                   done,
                   new_state,
                   memory,
                   gamma=0.99):
    if done:
      reward_sum = 0.  # terminal
    else:
      reward_sum = self.local_model(
          tf.convert_to_tensor(new_state[None, :],
                               dtype=tf.float32))[-1].numpy()[0]

    # Get discounted rewards
    disc_rewards = []
    for reward in memory.rewards[::-1]:  # reverse buffer r
      reward_sum = reward + gamma * reward_sum
      disc_rewards.append(reward_sum)
    disc_rewards.reverse()

    logits, values = self.local_model(
        tf.convert_to_tensor(np.vstack(memory.states),
                             dtype=tf.float32))
    # Get our advantages
    adv = tf.convert_to_tensor(np.array(disc_rewards)[:, None],
                            dtype=tf.float32) - values
    # Value loss
    val_loss = adv ** 2

    # Calculate our policy loss
    policy = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)

    pol_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                 logits=logits)
    pol_loss *= tf.stop_gradient(adv)
    pol_loss -= 0.01 * entropy
    total_loss = tf.reduce_mean((0.5 * val_loss + pol_loss))
    return total_loss


if __name__ == '__main__':
  print(args)
  agent = Agent()
  if args.train:
    agent.train()
  else:
    agent.play()