import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
from model import QFuncModel
from utils import *
from config import *

env = gym.make(args.game)
args.actions = env.action_space.n
model_target = QFuncModel(args)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('save')
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    env = wrappers.Monitor(env, 'play', force=True)
    for episode in range(args.num_play_episode):
        x_t = env.reset()
        x_t = rgb2gray(resize(x_t))
        s_t = np.stack([x_t for _ in range(args.frames)], axis=2)
        total_reward = 0
        while True:
            # env.render()
            a_t = np.zeros([args.actions])
            readout_t = sess.run(model_target.readout, feed_dict={model_target.s: [s_t]})
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
            x_t_next, r_t, terminal, info = env.step(action_index)
            total_reward += r_t
            if terminal:
                break
            else:
                x_t_next = rgb2gray(resize(x_t_next))
                s_t = np.append(x_t_next[:, :, np.newaxis], s_t[:, :, 0:3], axis=2)
        print "episode %d: score %d" % (episode, total_reward)

