# 可以使用ExperimentGrid 工具方便进行算法超参数的网格搜索，进而选择算法最优的超参数
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo
import tensorflow as tf

# TODO 还没有运行成功

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()
    eg = ExperimentGrid(name='ppo-bench')
    eg.add('env_name', 'CartPole-v0', '', True)  # eg.add(param_name, values, shorthand, in_name) in_name,
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
    eg.add('ac_kwargs:activation', [tf.tanh, tf.nn.relu], '')
    eg.run(ppo, num_cpu=args.cpu)
