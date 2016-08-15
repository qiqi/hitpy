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
from hit import run

cp_path = os.path.join(my_path, 'fds')
u0 = load(os.path.join(my_path, 'state_0.03.npy'))

mu = 0.03
k_modes = 16
m_segments = 100
steps_per_segment = 100
checkpoint = load_last_checkpoint(BASE_PATH, M_MODES)
if checkpoint is None:
    J, G = shadowing(run, u0, mu, k_modes, m_segments, steps_per_segment,
                     0, checkpoint_path=cp_path)
else:
    J, G = continue_shadowing(run, mu, k_modes, m_segments, steps_per_segment,
                     checkpoint_path=cp_path)
