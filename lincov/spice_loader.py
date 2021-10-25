import numpy as np

from spiceypy import spiceypy as spice
import spiceypy.utils.support_types as stypes

import os

class SpiceLoader(object):

    def __init__(self, mission = None, dir = 'kernels/', id = -5440):
        self.dir        = dir
        self.object_id  = id

        names = ['de432s.bsp', 'pck00010.tpc', 'naif0012.tls',
                 'moon_pa_de421_1900-2050.bpc', 'moon_080317.tf',
                 'moon_fixed_me.tf', 'gm_de431.tpc', 'earth_070425_370426_predict.bpc',
                 'earthstns_itrf93_050714.bsp', 'earth_topo_050714.tf']
        if mission is not None:
            names.append(mission + '.bsp')
            self.mission = mission

        self.load(names)

    def load(self, names):
        kernels_dir = os.path.join(os.path.dirname(__file__), '..', self.dir)
        
        paths = []
        for name in names:
            paths.append(kernels_dir + name)
            
        spice.furnsh(paths)

        self.loaded = paths

        # Pre-cache some things that we'll almost certainly need access to.
        self.load_constants()
        
        return self.loaded

    def load_constants(self):
        self.radii_earth     = np.array(spice.bodvcd(399, 'RADII', 3)[1])
        self.radii_moon      = np.array(spice.bodvcd(301, 'RADII', 3)[1])
        self.r_earth         = self.radii_earth[1]
        self.r_moon          = self.radii_moon[1]

        self.mu_earth        = spice.bodvcd(399, 'GM', 1)[1]
        self.mu_moon         = spice.bodvcd(301, 'GM', 1)[1]

        # Constants that should ultimately probably go in SPICE kernels
        # once we're working with attitude timelines and frame kernels.
        # For now let's just pretend they're in SPICE kernels since we
        # have no spacecraft attitude to contend with.
        self.T_body_to_att           = np.identity(3)
        self.T_body_to_horizon_cam   = np.identity(3)
        self.T_body_to_alt           = np.identity(3)
        self.T_body_to_vel           = np.identity(3)
        self.T_body_to_trn_cam       = np.identity(3)
        self.T_body_to_hrn_cam       = np.identity(3)
        
        self.start, self.end = self.coverage(id = self.object_id)

    def radii(self, body):
        if body in (399, 'earth'):
            return self.r_earth
        elif body in (301, 'moon'):
            return self.r_moon
        else:
            raise NotImplemented("body {} not pre-loaded".format(body))

    @classmethod
    def spk_coverage(self, path, id = -5440):
        """Get start and end ephemeris times for a SPK file."""
        coverage = stypes.SPICEDOUBLE_CELL(2)
        spice.spkcov(path, id, coverage)
        return spice.wnfetd(coverage, 0)

    def coverage(self, id = None):
        """Get mission coverage for a specific NAIF ID"""
        if id is None:
            id = self.object_id
        return SpiceLoader.spk_coverage(self.dir + self.mission + '.bsp', id)
