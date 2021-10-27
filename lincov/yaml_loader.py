import numpy as np
import pyquat as pq

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
yaml = YAML()


class AttributeDict(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def ax_char_to_unit_vector(ax):
    if ax == 'x':
        return np.array([1.0, 0.0, 0.0])
    elif ax == 'y':
        return np.array([0.0, 1.0, 0.0])
    elif ax == 'z':
        return np.array([0.0, 0.0, 1.0])
    else:
        raise ArgumentError("did not recognize axis '{}'".format(ax))


def direction_to_vector(pci, dir):
    """Given a planet-centered inertial frame and a direction string,
    return a vector in that inertial frame.
    
    Args:
        pci:  inertial state (position and velocity, length 6)
        dir:  string describing direction (one of nadir, zenith, 
              prograde, retrograde)

    Returns:
        A length 3 vector in the given inertial frame.
    """

    if dir == 'nadir':
        r = -np.array(pci[0:3])
        r /= linalg.norm(r)
    elif dir == 'zenith':
        r = np.array(pci[0:3])
        r /= linalg.norm(r)
    elif dir == 'prograde' or dir == 'retrograde':
        r = pci[0:3]
        v = pci[3:6]
        h = np.cross(r, v)
        prograde = np.cross(h, r)

        if dir == 'prograde':
            return prograde / linalg.norm(prograde)
        else:
            return -prograde / linalg.norm(prograde)


class AttitudeCommand(AttributeDict):
    def T_inrtl_to_body(self, state):
        if self.center == 'moon':
            pci = state.lci
        elif self.center == 'earth':
            pci = state.eci
        else:
            raise ArgumentError("center has support only for 'moon' or 'earth', found '{}'".format(self.center))

        # Get body frame vectors
        ub = np.array(self.primary['vector'])
        vb = np.array(self.secondary['vector'])

        # Get inertial frame vectors (e.g. nadir, zenith, etc.)
        u = direction_to_vector(self.primary['direction'])
        v = direction_to_vector(self.secondary['direction'])
        
        T_inrtl_to_body = pq.ref_obs_to_matrix(u, v, ub, vb)
        return T_inrtl_to_body


def scale(val, scale):
    if type(val) in (CommentedSeq, list, tuple):
        return np.array(val) * scale
    else:
        return val * scale
        
    
class YamlLoader(object):
    required_params = ('horizon_fov',
                       'horizon_max_phase_angle',
                       'radiometric_min_elevation')
                       
    
    def __init__(self, label):
        f = open("config/{}.yml".format(label), 'r')
        self.yaml = yaml.load(f)
        self.dt = self.yaml['dt']
        self.order = list(self.yaml['meas_dt'].keys())
        self.meas_dt = {}
        self.meas_last = {}
        self.block_dt = self.yaml['block_dt']
        
        for key in self.order:
            self.meas_dt[key] = self.yaml['meas_dt'][key]
            
        if 'meas_last' in self.yaml:
            for key in self.yaml['meas_last']:
                self.meas_last[key] = self.yaml['meas_last'][key]

        for key in self.order:
            if key not in self.meas_last:
                self.meas_last[key] = 0.0
                
        self.label = label
        self.params = AttributeDict(self.yaml['params'])
        if 'command' in self.yaml:
            self.command = AttitudeCommand(self.yaml['command'])
        else:
            self.command = None

        # Perform unit conversions for keys
        for key in self.yaml['params']:
            if key[-4:] == '_deg':
                self.params[key[:-4]] = scale(self.params[key], np.pi/180.0)
            elif key[-7:] == '_arcmin':
                self.params[key[:-7]] = scale(self.params[key], np.pi / (180*60.0))
            elif key[-7:] == '_arcsec':
                self.params[key[:-7]] = scale(self.params[key], np.pi / (180*3600.0))
            elif type(self.params[key]) == CommentedSeq:
                self.params[key] = np.array(self.params[key])
            elif key[-10:] == '_az_el_deg':
                # Start with an x unit vector.
                # Rotate it about the y axis to set the elevation.
                # Rotate it about the z axis to set the azimuth.
                # Also, convert from degrees to radians.
                az = scale(self.params[key][0], np.pi/180.0)
                el = scale(self.params[key][1], np.pi/180.0)
                dir = np.array([1.0, 0.0, 0.0])
                T_az_el = rotate_y(np.pi - el).dot(rotate_z(az)) # rotation from case frame to unit vector along z axis frame
                self.params['T_' + key[:-10]] = T_az_el

        # Set defaults for mandatory arguments. These are only here
        # because there are computations in State which happen
        # regardless of the measurement types that depend on these
        # parameters. State does not have access to the meas_dt dict.
        if 'horizon_earth' in self.meas_dt or 'horizon_moon' in self.meas_dt:
            if 'horizon_fov' not in self.params:
                print("Setting default: horizon_fov = 0")
                self.params.horizon_fov = 0.0
            if 'horizon_max_phase_angle' not in self.params:
                print("Setting default: horizon_max_phase_angle = 0")
                self.params.horizon_max_phase_angle = 0.0
        if 'twoway_range' in self.meas_dt or 'twoway_doppler' in self.meas_dt:
            if 'radiometric_min_elevation' not in self.params:
                print("Setting default: radiometric_min_elevation_deg = 90")
                self.params.radiometric_min_elevation = np.pi * 0.5 # 90 degrees
                
    def as_metadata(self):
        metadata              = self.yaml
        metadata['order']     = self.order
        metadata['meas_last'] = self.meas_last
        metadata['meas_dt']   = self.meas_dt
        return metadata
