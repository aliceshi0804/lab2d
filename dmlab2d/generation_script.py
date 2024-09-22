"""
Encapsulate generate data to make it parallel
"""
from os import makedirs
from os.path import join
import argparse
from multiprocessing import Pool
from subprocess import call

def _threaded_generation(args, rpt, i):
    tdir = join(args.rootdir, 'thread_{}'.format(i))
    makedirs(tdir, exist_ok=True)

    # Use xhost to allow connections to the X server
    # xhost_cmd = ['xhost', '+']
    # print(" ".join(xhost_cmd))
    # call(xhost_cmd, shell=True)

    # Set the display server number
    # display_cmd = ['export', 'DISPLAY=:{}'.format(i + 1)]
    # print(" ".join(display_cmd))
    # call(display_cmd, shell=True)

    # cmd = ['xhost', '+']

    # Run your Python script
    python_cmd = ['bazel', 'run', '-c',
                  'opt', 'random_agent', '--', '--level_name=chase_eat', '--num_episodes=' + str(args.rollouts)]
    print(" ".join(python_cmd))
    call(python_cmd, shell=True)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Total number of rollouts")
    parser.add_argument('--threads', type=int, help="Number of threads")
    parser.add_argument('--rootdir', type=str, help="Directory to store rollout "
                        "directories of each thread")
    parser.add_argument('--policy', type=str, choices=['brown', 'white'],
                        help="Directory to store rollout directories of each thread",
                        default='brown')
    args = parser.parse_args()

    rpt = args.rollouts // args.threads + 1

    with Pool(args.threads) as p:
        p.starmap(_threaded_generation,  [(args, rpt, i) for i in range(args.threads)])
