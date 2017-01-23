import tensorflow as tf
import numpy as np
import gym
import random
import threading
import time
import matplotlib.pyplot as plt
from collections import deque
from model import QFuncModel
from utils import *
from config import *

def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def train(threadid, sess, env):
    x_t = env.reset()
    x_t = rgb2gray(resize(x_t))
    s_t = np.stack([x_t for _ in range(args.frames)], axis=2)
    epsilon = args.initial_epsilon
    final_epsilon = sample_final_epsilon()
    t = 0
    total_reward = 0
    s_batch = []
    a_batch = []
    r_batch = []
    s_next_batch = []
    y_batch = []
    while True:
        a_t = np.zeros([args.actions])
        if random.random() < epsilon:
            action_index = random.randrange(args.actions)
        else:
            readout_t = sess.run(model.readout, feed_dict={model.s: [s_t]})
            action_index = np.argmax(readout_t)
        a_t[action_index] = 1
        x_t_next, r_t, terminal, info = env.step(action_index)
        x_t_next = rgb2gray(resize(x_t_next))
        s_t_next = np.append(x_t_next[:, :, np.newaxis], s_t[:, :, 0:3], axis=2)
        if epsilon > final_epsilon:
            epsilon -= (args.initial_epsilon - final_epsilon) / args.explore
        s_batch.append(s_t)
        a_batch.append(a_t)
        r_batch.append(r_t)
        s_next_batch.append(s_t_next)
        if t % args.async_target_update_freq == 0:
            print "thread %d update terget network, time %d" % (threadid, t)
            model_target.copy(sess, model)
        if t % args.batch_size == 0 or terminal:
            readout_batch = sess.run(model_target.readout, feed_dict={model_target.s: s_next_batch})
            for i in range(len(readout_batch) - 1):
                y_batch.append(r_batch[i] + args.gamma * np.max(readout_batch[i]))
            y_batch.append(r_batch[-1] if terminal else r_batch[-1] + args.gamma * np.max(readout_batch[-1]))
            sess.run(model.train_op, feed_dict={
                model.s: s_batch,
                model.y: y_batch,
                model.a: a_batch
            })
            s_batch = []
            a_batch = []
            r_batch = []
            s_next_batch = []
            y_batch = []
        #if threadid == 0:
            #env.render()
            #print "thread %d, time %d: action: %d, reward: %f" % (threadid, t, action_index, r_t)
        t += 1
        total_reward += r_t
        if terminal:
            print "thread %d game over, score %d, time %d" % (threadid, total_reward, t)
            total_reward = 0
            x_t = env.reset()
            x_t = rgb2gray(resize(x_t))
            s_t = np.stack([x_t for _ in range(args.frames)], axis=2)
        else:
            s_t = s_t_next
        if t % 10000 == 0 and threadid == 0:
            saver.save(sess, 'save/model.tfmodel', global_step=t)

threads = []
envs = [gym.make(args.game) for _ in range(args.async_thread_num)]
args.actions = envs[0].action_space.n
model_target = QFuncModel(args)
model = QFuncModel(args)
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    model_target.copy(sess, model)
    for i in range(args.async_thread_num):
        t = threading.Thread(target=train, args=(i, sess, envs[i]))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

