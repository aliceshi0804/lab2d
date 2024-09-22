import dmlab2d
from dmlab2d import runfiles_helper

lab = dmlab2d.Lab2d(runfiles_helper.find(), {"levelName": "chase_eat"})
env = dmlab2d.Environment(lab, ["WORLD.RGB"])
env.step({})