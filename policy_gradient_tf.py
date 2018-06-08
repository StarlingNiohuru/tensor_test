import itertools
import collections
import sys

import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class PolicyEstimator(object):
    def __init__(self, env, session=tf.Session(), learning_rate=0.01, latent_dim=10):
        self.env = env
        self.session = session
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.observation = None
        self.action = None
        self.value = None
        self.action_probs = None
        self.loss = None
        self.train_op = None

    def build_policy_estimator(self):
        with tf.variable_scope("policy_estimator"):
            self.observation = tf.placeholder(shape=(None, self.env.observation_space.shape[0]), dtype=tf.float32)
            self.action = tf.placeholder(shape=(None,), dtype=tf.int32)
            self.value = tf.placeholder(shape=(None,), dtype=tf.float32)

            hidden = tf.layers.dense(inputs=self.observation, units=self.latent_dim, activation=tf.nn.tanh,
                                     kernel_initializer=tf.random_normal_initializer())
            self.action_probs = tf.layers.dense(inputs=hidden, units=self.env.action_space.n, activation=tf.nn.softmax,
                                                kernel_initializer=tf.random_normal_initializer())

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.action_probs, labels=self.action)
            self.loss = tf.reduce_mean(cross_entropy * self.value)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

    def predict(self, observation):
        if observation.ndim == 1:
            observation = np.stack([observation])
        feed_dict = {self.observation: observation}
        return self.session.run(self.action_probs, feed_dict)

    def update(self, observation, action, value):
        if observation.ndim == 1:
            observation = np.stack([observation])
        if np.ndim(action) == 0:
            action = np.expand_dims(action, 0)
        if np.ndim(value) == 0:
            value = np.expand_dims(value, 0)
        feed_dict = {self.observation: observation, self.value: value, self.action: action}
        _, loss = self.session.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator(object):

    def __init__(self, env, session=tf.Session(), learning_rate=0.1, latent_dim=10):
        self.env = env
        self.session = session
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.observation = None
        self.target = None
        self.value_estimate = None
        self.loss = None
        self.train_op = None

    def build_value_estimator(self):
        with tf.variable_scope("value_estimator"):
            self.observation = tf.placeholder(shape=(None, self.env.observation_space.shape[0]), dtype=tf.float32)
            self.target = tf.placeholder(shape=(None,), dtype=tf.float32)

            hidden = tf.layers.dense(inputs=self.observation, units=self.latent_dim, activation=tf.nn.tanh,
                                     kernel_initializer=tf.random_normal_initializer())
            output = tf.layers.dense(inputs=hidden, units=1,
                                     kernel_initializer=tf.random_normal_initializer())
            self.value_estimate = tf.squeeze(output)

            self.loss = tf.reduce_mean(tf.squared_difference(self.value_estimate, self.target))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

    def predict(self, observation):
        if observation.ndim == 1:
            observation = np.stack([observation])
        feed_dict = {self.observation: observation}
        return self.session.run(self.value_estimate, feed_dict)

    def update(self, observation, target):
        if observation.ndim == 1:
            observation = np.stack([observation])
        if np.ndim(target) == 0:
            target = np.expand_dims(target, 0)
        feed_dict = {self.observation: observation, self.target: target}
        _, loss = self.session.run([self.train_op, self.loss], feed_dict)
        return loss


def plot_episode_stats(stats, output_dir):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    fig1.savefig(output_dir + '/fig1.png', dpi=fig1.dpi)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Rewards")
    plt.title("Episode Reward over Time")
    fig2.savefig(output_dir + '/fig2.png', dpi=fig2.dpi)
    return fig1, fig2


def actor_critic(env, policy_estimator, value_estimator, num_episodes, discount_factor=0.9, render=False):
    EpisodeStats = collections.namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards"])
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
    Transition = collections.namedtuple("Transition", ["observation", "action", "reward", "next_observation", "done"])

    for i_episode in range(num_episodes):
        observation = env.reset()
        episode = []
        for step in itertools.count():
            if render:
                env.render()
            action_probs = policy_estimator.predict(observation)[0]
            action = np.random.choice(range(len(action_probs)), p=action_probs)
            next_observation, reward, done, _ = env.step(action)

            episode.append(Transition(
                observation=observation, action=action, reward=reward, next_observation=next_observation, done=done))

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = step

            # Calculate TD Target
            value_next = value_estimator.predict(next_observation)
            td_target = reward + discount_factor * value_next
            td_error = td_target - value_estimator.predict(observation)
            value_estimator.update(observation, td_target)
            policy_estimator.update(observation, action, td_error)

            print("Step {} of Episode {}/{} (Rewards {})".format(step, i_episode + 1, num_episodes,
                                                                 stats.episode_rewards[i_episode]))
            if done:
                break
            observation = next_observation

    return stats


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    # env = gym.make('MountainCar-v0')
    if sys.argv[1] == 'train':
        with tf.Session() as sess:
            pe = PolicyEstimator(env=env, session=sess)
            pe.build_policy_estimator()
            ve = ValueEstimator(env=env, session=sess)
            ve.build_value_estimator()
            sess.run(tf.global_variables_initializer())
            stats = actor_critic(env, pe, ve, 10000)
            plot_episode_stats(stats, 'D:\deep_learning\samples')
    elif sys.argv[1] == 'test':
        env.reset()
        olist = []
        rlist = []
        alist = []
        for _ in range(10):
            a = env.action_space.sample()
            alist.append(a)
            o, r, _, _ = env.step(a)
            olist.append(o)
            rlist.append(r)
        observation = np.stack(olist)
        action = np.stack(alist)
        reward = np.stack(rlist)
        print(action)
        print(reward)
        with tf.Session() as sess:
            pe = PolicyEstimator(env=env, session=sess)
            pe.build_policy_estimator()
            sess.run(tf.global_variables_initializer())
            loss = pe.update(observation, action, reward)
            action_probs = pe.predict(observation)
            print(loss)
            print(range(len(action_probs[0])))
            ve = ValueEstimator(env=env, session=sess)
            ve.build_value_estimator()
            sess.run(tf.global_variables_initializer())
            loss = ve.update(observation, reward)
            value = ve.predict(observation)
            print(loss)
            print(value.shape)
