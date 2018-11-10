from spinup import ppo
import tensorflow as tf
import gym

if __name__ == '__main__':
    env_fn = lambda: gym.make('LunarLander-v2')

    ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)  # 设定算法参数

    logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')  # 设置日志输出位置

    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=10, logger_kwargs=logger_kwargs)

    # 使用 python -m spinup.run test_policy path/to/output_directory  可视化训练过程
    # 使用 python -m spinup.run plot [path/to/output_directory/] [--legend [LEGEND ...]]
    #     [--xaxis XAXIS] [--value [VALUE ...]] [--count] [--smooth S]
    #     [--select [SEL ...]] [--exclude [EXC ...]]
    # 绘图结果变化趋势
