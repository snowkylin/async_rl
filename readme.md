# Play Atari Games with TensorFlow and Asynchronous RL

This is a Tensorflow implementation of asyncronous 1-step Q learning with improvement on weight update process (use minibatch) to speed up the training process. Algorithm can be fount at [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

## Demo

[![Play Flappy Bird with TensorFlow ](https://img.youtube.com/vi/ZxHAf5BM0QM/0.jpg)](https://www.youtube.com/watch?v=ZxHAf5BM0QM)

## Dependencies
* Python
* TensorFlow
* gym (with atari environment)
* OpenCV-Python

## Usage
Run `play.py` to play atari game (default is Breakout-v0) by trained network.

Run `train.py` to train the network on your computer.

You will get a comparatively good result (40+ score) when t is larger than 2000000. On my computer (i5-4590/16GB/GTX 1060 6GB), the training process need at least 2-3 hours.

## Credit
* [coreylynch/async-rl](https://github.com/coreylynch/async-rl)
* [yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)

