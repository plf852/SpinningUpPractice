from spinup import ppo
import tensorflow as tf
import gym

if __name__ == '__main__':
    env_fn = lambda: gym.make('LunarLander-v2')

    ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)  # 设定算法参数

    logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')  # 设置日志输出位置

    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=10, logger_kwargs=logger_kwargs)