import numpy as np
import numpy.linalg as npl

import pyquat as pq

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


def lvlh_direction_to_vector(x, desc, y = None):
    """Given a planet-centered inertial frame and a direction string,
    return a vector in that inertial frame.
    
    Args:
        x:     state in whatever frame we like (length 6)
        desc:  string describing direction (one of nadir, zenith, 
               prograde, retrograde)

    Returns:
        A length 3 vector in the given inertial frame.
    """
    if len(x) == 6:
        r = x[0:3]
        v = x[3:6]
    else:
        r = x
        v = y
    if desc in ('zenith', 'radial'):
        return r / npl.norm(r)
    elif desc == 'nadir':
        return -r / npl.norm(r)
    elif desc in ('prograde', 'retrograde'):
        h = np.cross(r, v)
        h /= npl.norm(h)
        horiz = np.cross(h, r)
        u = horiz / npl.norm(horiz)
        if desc == 'retrograde':
            u = -u
        return u
    else:
        raise ValueError("unrecognized LVLH direction descriptor")

def command_to_vector(u,desc):
    if desc == 'nadir':
        x = np.array([-1.0, 0.0, 0.0 ])
    elif desc == 'zenith':
        x = np.array([ 1.0, 0.0, 0.0 ])
    elif desc == 'prograde':
        x = np.array([0.0, 1.0, 0.0])
    elif desc == 'retrograde':
        x = np.array([0.0, -1.0, 0.0])
    elif desc == 'crosstrack':
        x = np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError("unrecognized direction descriptor")
    return x


class AttitudeCommand(AttributeDict):
    def T_inrtl_to_body(self, state):
        if self.center == 'moon':
            pci = state.lci
        elif self.center == 'earth':
            pci = state.eci
        else:
            raise ValueError("center has support only for 'moon' or 'earth', found '{}'".format(self.center))

        ub = np.array(self.primary['vector'])
        vb = np.array(self.secondary['vector'])
        T_local_to_body = pq.uv_to_matrix(ub, vb).T

        # Get inertial to local coordinate system, which is primary.direction by secondary.direction
        ul = lvlh_direction_to_vector(pci, self.primary['direction'])
        vl = lvlh_direction_to_vector(pci, self.secondary['direction'])

        T_inrtl_to_local = pq.uv_to_matrix(ul, vl)

        return T_local_to_body.dot(T_inrtl_to_local)