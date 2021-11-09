#!/usr/bin/env python3

import numpy as np
from spiceypy import spiceypy as sp

import os.path
import sys

if __name__ == '__main__':

    # Need leapseconds kernel to output UTC
    sp.furnsh("kernels/naif0012.tls")

    # Argument is the name of a mission segment
    segment = sys.argv[1]
    try:
        count = int(sys.argv[2])
        et = np.load(os.path.join("output", segment, "time.{:04d}.npy".format(count))).item()
    except ValueError:
        label = sys.argv[2]
        et = np.load(os.path.join("output", segment, "time.{}.npy".format(label))).item()
    utc = sp.et2utc(et, 'C', 4)
    print("ephemeris time = {}".format(et))
    print("UTC            = {}".format(utc))
