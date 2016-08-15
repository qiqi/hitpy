import os
import sys
import shutil
import tempfile
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(my_path)
sys.path.append(os.path.join(my_path, '..', 'fds'))

from fds import *
from fds.checkpoint import load_last_checkpoint
from hit import run

cp_path = os.path.join(my_path, 'fds')
if not os.path.exists(cp_path):
    os.mkdir(cp_path)
u0 = load(os.path.join(my_path, 'state_0.05.npy'))

mu = 0.05
k_modes = 180
m_segments = 500
steps_per_segment = 200
checkpoint = load_last_checkpoint(cp_path, k_modes)
if checkpoint is None:
    J, G = shadowing(run, u0, mu, k_modes, m_segments, steps_per_segment,
                     0, checkpoint_path=cp_path)
else:
    J, G = continue_shadowing(run, mu, k_modes, m_segments, steps_per_segment,
                     checkpoint_path=cp_path)
