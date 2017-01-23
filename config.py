import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='Breakout-v0',
                    help='Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
parser.add_argument('--observe', type=int, default=10000,
                    help='timesteps to observe before training')
parser.add_argument('--explore', type=int, default=100000,
                    help='frames over which to anneal epsilon')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='decay rate of past observations')
parser.add_argument('--initial_epsilon', type=float, default=1.0,
                    help='starting value of epsilon')
parser.add_argument('--final_epsilon', type=float, default=0.0001,
                    help='final value of epsilon')
parser.add_argument('--frame_per_action', type=int, default=1,
                    help='')
parser.add_argument('--replay_memory', type=int, default=50000,
                    help='')
parser.add_argument('--resize_width', type=int, default=80,
                    help='')
parser.add_argument('--resize_height', type=int, default=80,
                    help='')
parser.add_argument('--frames', type=int, default=4,
                    help='')
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')
parser.add_argument('--learning_rate', type=float, default=0.000001,
                    help='learning rate')
parser.add_argument('--async_thread_num', type=int, default=8,
                    help='async_thread_num')
parser.add_argument('--async_target_update_freq', type=int, default=1000,
                    help='async_thread_num')
args = parser.parse_args()