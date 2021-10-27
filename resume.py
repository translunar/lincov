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

    label = sys.argv[1]    
    mission = sys.argv[2]
    object_id = int(sys.argv[3])
    count = int(sys.argv[4])
    loader = SpiceLoader(mission, id = object_id)

    l = LinCov.start_from(loader, label, count = count)
    while not l.finished:
        for step, mode in l.run():
            progress_bar(60, step.time - loader.start, loader.end - loader.start)


