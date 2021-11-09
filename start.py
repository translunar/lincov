#!/usr/bin/env python3

import numpy as np
from scipy.linalg import norm

from lincov import LinCov, progress_bar
from lincov.spice_loader import SpiceLoader
from lincov.yaml_loader import YamlLoader

import sys
import os

if __name__ == '__main__':

    np.seterr(invalid='raise')

    if len(sys.argv) < 2:
        raise SyntaxError("expected mission and segment label")

    mission = sys.argv[1]
    label = sys.argv[2]
    config = YamlLoader(mission, label)
    loader = SpiceLoader(mission, id = config.object_id)

    if len(sys.argv) > 3:
        copy_from = sys.argv[3]
    else:
        copy_from = 'f9'

    if len(sys.argv) > 4:
        snapshot_label = sys.argv[4]
    else:
        snapshot_label = 'init'

    if os.path.exists(os.path.join("output", label)):
        raise IOError("output directory already exists")

    l = LinCov.start_from(loader, label, copy_from = copy_from, snapshot_label = snapshot_label)
    while not l.finished:
        for step, mode in l.run():
            progress_bar(60, step.time - config.time.start, config.time.end - config.time.start)
    


    
