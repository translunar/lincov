from spiceypy import spiceypy as spice
import spiceypy.utils.exceptions as spexc

import numpy as np
from scipy.linalg import norm

import lincov.horizon as horizon


def sun_spacecraft_angle(body, time, object_id):
    if body == 'earth':
        frame = 'ITRF93'
    elif body == 'moon':
        frame = 'MOON_ME'
    sun,t1,t2 = spice.subslr('INTERCEPT/ELLIPSOID', body, time, frame, 'NONE', str(object_id))
    sc, t1,t2  = spice.subpnt('INTERCEPT/ELLIPSOID', body, time, frame, 'NONE', str(object_id))
    sun /= norm(sun)
    sc /= norm(sc)
    return np.arccos(np.dot(sun, sc))

def altitude_along_vector(loader, body, position, boresight):
    if body == 'earth':
        radii = loader.radii_earth
    elif body == 'moon':
        radii = loader.radii_moon
    
    if np.dot(boresight, position) >= 0:
        return float('inf')

    # Does this need to check for within planetary ellipsoid?

    nearest_point, distance = spice.npedln(*radii, position, boresight)
    if distance == 0.0:
        return norm(position - nearest_point)
    else:
        return float('inf')
    

class State(object):
    """State information for the linear covariance analysis"""

    r_station_ecef = {}
    
    def __init__(self, time, loader = None, params = None, command = None):
        self.loader = loader
        self.params = params
        self.command = command
        self.mu_earth = loader.mu_earth * 1e9 # put in m^3/s^2
        self.mu_moon  = loader.mu_moon * 1e9

        # Ensure that ground station locations are loaded
        if len(State.r_station_ecef) == 0 and (self.params is not None and 'ground_stations' in self.params):
            for station in self.params.ground_stations:
                gid = self.params.ground_stations[station]
                State.r_station_ecef[station] = spice.spkezr(station, self.loader.start, 'ITRF93', 'NONE', 'earth')[0][0:3] * 1000.0

        # Get planet-centered inertial state
        self.eci = spice.spkez(loader.object_id, time, 'J2000', 'NONE', 399)[0] * 1000.0
        self.lci = spice.spkez(loader.object_id, time, 'J2000', 'NONE', 301)[0] * 1000.0

        if command: # use the user-specified attitude command in the YML
            self.T_inrtl_to_body = self.command.T_inrtl_to_body(self)
        else: # no attitude commanded, use identity
            self.T_inrtl_to_body = np.identity(3)
        
        self.T_inrtl_to_lclf = spice.sxform('J2000', 'MOON_ME', time)
        self.T_inrtl_to_ecef = spice.sxform('J2000', 'ITRF93', time)
        self.T_lclf_to_body  = self.T_inrtl_to_body.dot(self.T_inrtl_to_lclf[0:3,0:3].T)
        self.T_ecef_to_body  = self.T_inrtl_to_body.dot(self.T_inrtl_to_ecef[0:3,0:3].T)

        # Get planet-fixed state
        self.ecef = self.T_inrtl_to_ecef.dot(self.eci)
        self.lclf = self.T_inrtl_to_lclf.dot(self.lci)

        # FIXME: Need measurements here
        self.a_meas_inrtl = np.zeros(3)
        self.w_meas_inrtl = np.zeros(3)

        # Get distance to earth and moon
        self.d_earth = norm(self.eci[0:3])
        self.d_moon  = norm(self.lci[0:3])

        # Get angular size of each
        self.earth_angle = 2 * np.arctan(self.loader.r_earth * 1000.0 / self.d_earth)
        self.moon_angle  = 2 * np.arctan(self.loader.r_moon * 1000.0 / self.d_moon)
        
        self.earth_phase_angle = sun_spacecraft_angle('earth', time, loader.object_id)
        self.moon_phase_angle  = sun_spacecraft_angle('moon', time, loader.object_id)

        # We need to be able to clearly see the planet in order to do
        # horizon detection.
        try:
            planet_occult_code = spice.occult('earth', 'ellipsoid', 'ITRF93', 'moon', 'ellipsoid', 'MOON_ME', 'NONE', str(loader.object_id), time)
        except spexc.SpiceNOTDISJOINT:
            planet_occult_code = None
            moon_occult_code = None

        self.horizon_moon_enabled  = False
        self.horizon_earth_enabled = False
        self.velocimeter_enabled   = False
        self.altimeter_enabled     = False
        
        if planet_occult_code == 0:
            if 'horizon_fov' in self.params:
                if self.earth_angle < self.params.horizon_fov and self.earth_phase_angle < self.params.horizon_max_phase_angle:
                    self.horizon_earth_enabled = True
            
                if self.moon_angle < self.params.horizon_fov and self.moon_phase_angle < self.params.horizon_max_phase_angle:
                    self.horizon_moon_enabled = True
        else:
            self.earth_angle = 0.0
            self.moon_angle  = 0.0

        self.elevation_from = {}
        self.visible_from = []
        self.r_station_inrtl = {}
        for ground_name in self.r_station_ecef:
            obj_str = str(self.loader.object_id)

            try:
                moon_occult_code = spice.occult(obj_str, 'point', ' ', 'moon', 'ellipsoid', 'MOON_ME', 'NONE', str(self.params.ground_stations[ground_name]), time)
            except spexc.SpiceNOTDISJOINT:
                pass # Already handled

            elevation = float('nan')
            if moon_occult_code is not None and moon_occult_code >= 0:
                # get spacecraft elevation
                x, lt = spice.spkcpo(obj_str, time, ground_name + '_TOPO', 'OBSERVER', 'NONE', self.r_station_ecef[ground_name] / 1000.0, 'earth', 'ITRF93')
                r, lon, lat = spice.reclat(x[0:3])

                if lat >= self.params.radiometric_min_elevation:
                    elevation = lat
                    self.visible_from.append(ground_name)
                    
            # store elevation of spacecraft for logging purposes
            self.elevation_from[ground_name] = elevation

        self.range_earth = norm(self.eci[0:3])
        self.range_moon  = norm(self.lci[0:3])

        # Get boresight directions for velocimeter and altimeter so we can take measurements with them.
        T_alt_to_inrtl = self.T_inrtl_to_body.T.dot(self.T_alt_to_body)
        T_vel_to_inrtl = self.T_inrtl_to_body.T.dot(self.T_vel_to_body)
        altimeter_boresight_inrtl  = T_alt_to_inrtl.dot(np.array([0.0, 0.0, 1.0]))
        altimeter_boresight_inrtl /= norm(alt_boresight_inrtl)
        velocimeter_boresight_inrtl  = T_vel_to_inrtl.dot(np.array([0.0, 0.0, 1.0]))
        velocimeter_boresight_inrtl /= norm(vel_boresight_inrtl)

        # Determine altitudes along boresights of altimeter and velocimeter (initially for comparison to range constraints
        # on these sensors)
        self.alt_altimeter_earth   = altitude_along_vector(self.loader, 'earth', self.eci, altimeter_boresight_inrtl)
        self.alt_altimeter_moon    = altitude_along_vector(self.loader, 'moon',  self.lci, altimeter_boresight_inrtl)
        self.alt_velocimeter_earth = altitude_along_vector(self.loader, 'earth', self.eci, velocimeter_boresight_inrtl)
        self.alt_velocimeter_moon  = altitude_along_vector(self.loader, 'moon',  self.lci, velocimeter_boresight_inrtl)

        self.time = time

    @property
    def object_id(self):
        return self.loader.object_id

    @property
    def T_body_to_att(self):
        return self.loader.T_body_to_att

    @property
    def T_body_to_horizon_cam(self):
        return self.loader.T_body_to_horizon_cam

    @property
    def T_body_to_trn_cam(self):
        return self.loader.T_body_to_trn_cam

    @property
    def T_body_to_hrn_cam(self):
        return self.loader.T_body_to_hrn_cam

    @property
    def T_body_to_alt(self):
        return self.loader.T_body_to_alt

    @property
    def T_alt_to_body(self):
        return self.loader.T_body_to_alt.T

    @property
    def T_body_to_vel(self):
        return self.loader.T_body_to_vel

    @property
    def T_vel_to_body(self):
        return self.loader.T_body_to_vel.T

    def range(self, rel):
        if rel == 'earth':
            return self.range_earth
        elif rel == 'moon':
            return self.range_moon

    def radii(self, rel):
        if rel == 'earth': return self.loader.r_earth * 1000.0
        elif rel == 'moon': return self.loader.self.r_moon * 1000.0

    def T_pa_to_cam(self, rel):
        """Transformation from cone principal axis frame to opnav camera
        frame, where rel tells which planet cone is oriented
        towards.

        """
        if rel in ('earth', 399):
            body_id = 399
        elif rel in ('moon', 301):
            body_id = 301
        else:
            raise NotImplemented("expected 'moon' or 'earth' based opnav, got '{}'".format(rel))
        return horizon.compute_T_pa_to_cam(self.eci, self.time, body_id)

    @property
    def T_epa_to_cam(self):
        return self.T_pa_to_cam('earth')

    @property
    def T_mpa_to_cam(self):
        return self.T_pa_to_cam('moon')
