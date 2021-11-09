#!/usr/bin/env python3

import numpy as np
from scipy.linalg import norm

from lincov import LinCov, progress_bar
from lincov.spice_loader import SpiceLoader
from lincov.yaml_loader import YamlLoader

import sys


if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise SyntaxError("expected run name")
    
    mission = sys.argv[1]
    label = sys.argv[2]
    config = YamlLoader(mission, label)
    count = int(sys.argv[3])
    loader = SpiceLoader(mission, id = config.object_id)

    l = LinCov.start_from(loader, label, count = count)
    while not l.finished:
        for step, mode in l.run():
            progress_bar(60, step.time - config.time.start, config.time.end - config.time.start)


