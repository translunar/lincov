import numpy as np
import numpy.linalg as npl

import pyquat as pq

from lincov.frames import rotate_x, rotate_y, rotate_z
from lincov.attitude_command import AttitudeCommand, AttributeDict
from lincov.spice_loader import SpiceLoader
from spiceypy import spiceypy as sp

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
yaml = YAML()


def scale(val, scale):
    if type(val) in (CommentedSeq, list, tuple):
        return np.array(val) * scale
    else:
        return val * scale


def convert_key_value_shorthand(key, value, accum):
    """Handle unit conversions and other sorts of short-hands.
    
    Args:
        key:    some config file key
        value:  some corresponding value to convert
        accum:  the new dict we're building

    Returns:
        This method returns the dict accum with the additional keys and values.
    """
    accum[key] = value # retain the old value too
    if key[-10:] == '_az_el_deg':
        # Start with a z unit vector.
        # Rotate it about the y axis to set the elevation.
        # Rotate it about the z axis to set the azimuth.
        # Also, convert from degrees to radians.        
        az = scale(value[0], np.pi/180.0)
        el = scale(value[1], np.pi/180.0)
        T_az_el = rotate_y(el).dot(rotate_z(az)) # rotation from case frame to unit vector along z axis frame
        T_az_el[np.where(np.abs(T_az_el) < 1e-15)] = 0.0 # remove small quantities
        accum['T_' + key[:-10]] = T_az_el
    elif key[-4:] == '_deg':
        accum[key[:-4]] = scale(value, np.pi / 180.0)
    elif key[-7:] == '_arcmin':
        accum[key[:-7]] = scale(value, np.pi / (180.0 * 60.0))
    elif key[-7:] == '_arcsec':
        accum[key[:-7]] = scale(value, np.pi / (180.0 * 3600.0))
    elif type(value) == CommentedSeq:
        accum[key] = np.array(value)
    return accum
    
class YamlLoader(object):
    required_params = ('horizon_fov',
                       'horizon_max_phase_angle',
                       'radiometric_min_elevation')
                       
    
    def __init__(self, mission, label):
        f = open("config/{}.yml".format(mission), 'r')
        self.yaml = yaml.load(f)
        self.config = self.yaml[label]
        self.object_id = int(self.yaml['object_id'])

        # Now we can load the SPICE files we need
        self.spice = SpiceLoader(mission, id = self.object_id)

        self.time       = AttributeDict(self.config['time'])
        self.time.order = list(self.time.meas_dt.keys())
        self.time.start = sp.str2et(self.time.start)
        self.time.end   = sp.str2et(self.time.end)

        replace_meas_dt = {}
        for key in self.time.order:
            replace_meas_dt[key] = self.time.meas_dt[key]
        self.time.meas_dt = replace_meas_dt

        if 'meas_last' not in self.time:
            self.time['meas_last'] = AttributeDict({})

        for key in self.time.order:
            if key not in self.time.meas_last:
                self.time.meas_last[key] = 0.0
                
        self.label = label
        params_dict = {}
        for key in self.config['params']:
            params_dict = convert_key_value_shorthand(key, self.config['params'][key], params_dict)
        self.params = AttributeDict(params_dict)

        self.params.noise = AttributeDict(self.params.noise)
        self.params.sensors = AttributeDict(self.params.sensors)

        if 'command' in self.config:
            self.command = AttitudeCommand(self.config['command'])
        else:
            self.command = None

        # Perform unit conversions for keys
        for sensor in self.params.sensors:
            sensor_dict = {}
            for key in self.params.sensors[sensor]:
                sensor_dict = convert_key_value_shorthand(key, self.params.sensors[sensor][key], sensor_dict)
            self.params.sensors[sensor] = AttributeDict(sensor_dict)
                

        # Set defaults for mandatory arguments. These are only here
        # because there are computations in State which happen
        # regardless of the measurement types that depend on these
        # parameters. State does not have access to the meas_dt dict.
        if 'horizon_earth' in self.time.meas_dt or 'horizon_moon' in self.time.meas_dt:
            if 'horizon' not in self.params.sensors:
                self.params.sensors['horizon'] = {}
            if 'fov' not in self.params.sensors['horizon']:
                print("Setting sensor default: horizon.fov = 0")
                self.params.sensors.horizon.fov = 0.0
            if 'max_phase_angle' not in self.params.sensors['horizon']:
                print("Setting sensor default: horizon.max_phase_angle = 0")
                self.params.horizon_max_phase_angle = 0.0
        if 'twoway_range' in self.time.meas_dt or 'twoway_doppler' in self.time.meas_dt:
            if 'radiometric_min_elevation' not in self.params:
                import pdb
                pdb.set_trace()
                print("Setting default: radiometric_min_elevation_deg = 90")
                self.params.radiometric_min_elevation = np.pi * 0.5 # 90 degrees

        if 'ground_stations' not in self.params:
            self.params['ground_stations'] = {}
                
    def as_metadata(self):
        metadata              = self.yaml
        metadata['order']     = self.time.order
        metadata['meas_last'] = self.time.meas_last
        metadata['meas_dt']   = self.time.meas_dt
        return metadata

    def find_count(self, et):
        """Given some ephemeris time, what index number should we look in to
        find it?
        
        Args:
            et:         ephemeris time in seconds
            
        Returns:
            A file index (integer).
        """
        return int(np.floor((et - self.time.start) / self.time.block_dt)) - 1