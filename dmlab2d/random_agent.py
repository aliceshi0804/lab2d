# Copyright 2019 The DMLab2D Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Random agent for running against DM Lab2D environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

import numpy as np
import pygame

import dmlab2d
from dmlab2d import runfiles_helper

import os
from os import makedirs
from os.path import join

def _make_int32_distribution(random, minimum, maximum):

  def function():
    return random.randint(minimum, maximum + 1)

  return function


def _make_float64_distribution(random, minimum, maximum):

  def function():
    return random.uniform(minimum, maximum)

  return function


class PyGameRandomAgent(object):
  """Random agent works with int32 or float64 bounded actions."""

  def __init__(self, action_spec, observation_name, observation_spec, seed,
               scale):
    """Create a PyGame agent.

    Args:
      action_spec: Environment action spec used to generate random actions.
      observation_name: Name of observation to render each frame.
      observation_spec: Environment observation spec for creating PyGame window.
      seed: Agent seed used for generating random actions.
      scale: Scales screen.
    """
    self._observation_name = observation_name
    random = np.random.RandomState(seed)
    self._actions = []
    self._scores = []
    self._scale = scale
    for name, spec in action_spec.items():
      if spec.dtype == np.dtype('int32'):
        self._actions.append(
            (name, _make_int32_distribution(random, spec.minimum,
                                            spec.maximum)))
      elif spec.dtype == np.dtype('float64'):
        self._actions.append(
            (name, _make_float64_distribution(random, spec.minimum,
                                              spec.maximum)))
      else:
        print("Warning '{}' is not supported".format(spec))

  def step(self, timestep):
    """Renders timestep and returns random actions according to spec."""
    if timestep.reward is not None:
      if timestep.reward != 0:
        self._scores[-1] += timestep.reward
        display_score_dirty = True
    else:
      self._scores.append(0)
      display_score_dirty = True
    return {name: gen() for name, gen in self._actions}

  def print_stats(self):
    print('Scores: ' + ', '.join(str(score) for score in self._scores))


def _create_environment(args):
  """Creates an environment.

  Args:
    args: See `main()` for description of args.

  Returns:
    dmlab2d.Environment with one observation.
  """
  # print("created environment")
  args.settings['levelName'] = args.level_name
  lab2d = dmlab2d.Lab2d(runfiles_helper.find(), args.settings)
  return dmlab2d.Environment(lab2d, [args.observation], args.env_seed)


def _run(args):
  """Runs a random agent against an environment rendering the results.

  Args:
    args: See `main()` for description of args.
  """
  # print("running")
  env = _create_environment(args)
  agent = PyGameRandomAgent(env.action_spec(), args.observation,
                            env.observation_spec(), args.agent_seed, args.scale)
  for i in range(args.num_episodes):
    timestep = env.reset()
    # Run single episode.
    s_rollout = []
    r_rollout = []
    d_rollout = []
    a_rollout = []
    t_rollout = []

    enough_data = False
    while not enough_data:
      action = agent.step(timestep)
      timestep = env.step(action)
      s_rollout += [timestep.observation['WORLD.RGB']] # observation
      r_rollout += [timestep.reward] # reward
      d_rollout += [timestep.discount] # discount
      a_rollout += [action["MOVE"]] # actions
      t_rollout += [timestep.step_type.value] # terminals
      
      if timestep.last():
        # Observe last frame of episode.
        agent.step(timestep)
        a_rollout += [action["MOVE"]]

        if len(a_rollout) <= 1000:
          # print("not enough data")
          timestep = env.reset()
          # Run single episode.
          s_rollout = []
          r_rollout = []
          d_rollout = []
          a_rollout = []
          t_rollout = []
        else:
          enough_data = True
          print("> End of rollout {}, {} frames...".format(i, args.num_episodes))
          # print(s_rollout, r_rollout, d_rollout, a_rollout, t_rollout)
          makedirs("/Users/qingshi/docs/Projects/sp24/lab2d/chase_eat_may8/thread_" + str(args.env_seed), exist_ok=True)
          np.savez(join("/Users/qingshi/docs/Projects/sp24/lab2d/chase_eat_may8/thread_{}".format(args.env_seed), 'rollout_{}'.format(i)),
                  observations=np.array(s_rollout),
                  rewards=np.array(r_rollout),
                  discounts=np.array(d_rollout),
                  actions=np.array(a_rollout),
                  terminals=np.array(t_rollout))
          file_path = os.path.abspath(join("/Users/qingshi/docs/Projects/sp24/lab2d/chase_eat_may8/thread_" + str(args.env_seed), 'rollout_{}.npz'.format(i)))
          print("File will be saved to:", file_path)
          break
  
  print("All episodes completed, report per episode.")
  # All episodes completed, report per episode.
  agent.print_stats()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name', type=str, default='clean_up', help='Level name to load')
  parser.add_argument(
      '--observation',
      type=str,
      default='WORLD.RGB',
      help='Observation to render')
  parser.add_argument(
      '--settings', type=json.loads, default={}, help='Settings as JSON string')
  parser.add_argument(
      '--env_seed', type=int, default=0, help='Environment seed')
  parser.add_argument('--agent_seed', type=int, default=0, help='Agent seed')
  parser.add_argument(
      '--num_episodes', type=int, default=1, help='Number of episodes')
  parser.add_argument(
      '--scale', type=float, default=1, help='Scale to render screen')

  args = parser.parse_args()
  _run(args)


if __name__ == '__main__':
  main()
