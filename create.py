#!/usr/bin/env python3
"""This script creates a new initial state. It's mainly an example, 
using a Falcon 9 launch as the starting point."""

import numpy as np
from scipy.linalg import norm

from lincov import LinCov, progress_bar
from lincov.spice_loader import SpiceLoader
from lincov.yaml_loader import YamlLoader
from lincov.launch import sample_f9_gto_covariance
from lincov.state import State

from spiceypy import spiceypy as spice

import sys
import os

if __name__ =='__main__':

    mission = sys.argv[1]
    object_id = int(sys.argv[2])
    loader = SpiceLoader(mission, id = object_id)

    time = loader.start

    x = State(time, loader)

    P = sample_f9_gto_covariance(x)
    LinCov.save_covariance('f9', P, time, snapshot_label = 'f9')
