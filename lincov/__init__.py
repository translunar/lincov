import numpy as np
from scipy.linalg import norm, inv

from lincov.spice_loader import SpiceLoader
from spiceypy import spiceypy as spice

from lincov.state import State
import lincov.horizon as horizon
from lincov.launch import sample_f9_gto_covariance
from lincov.frames import compute_T_inrtl_to_lvlh
from lincov.gravity import gradient as G
from lincov.yaml_loader import YamlLoader
from lincov.reader import find_block

import pyquat as pq
import pandas as pd

import pathlib

import math

import pickle


def progress_bar(bar_length, completed, total):
    # https://stackoverflow.com/a/50108192/170300
    bar_length_unit_value = (total / bar_length)
    completed_bar_part = math.ceil(completed / bar_length_unit_value)
    progress = "*" * completed_bar_part
    remaining = " " * (bar_length - completed_bar_part)
    percent_done = "%.2f" % ((completed / total) * 100)
    print(f'[{progress}{remaining}] {percent_done}%', end='\r')
    

class LinCov(object):
    """This object is responsible for managing instances of linear
    covariance analysis.

    A LinCov object has the following instance variables, which it
    must either load from a previous instance or get from arguments:

    * dt        propagation time step (s)
    * start     initial time (seconds since J2000)
    * end       final time (by default, start + self.block_dt)
    * meas_dt   update time step dict (for each update type)
    * meas_last tracker for last time an update occurred
    * count     index number for the save files produced by this run
    * block_dt  the duration of a single run (and the time length of
                save files), which should be constant across all
                runs in a given label
    * label     refers to a given set of test conditions

    """
    N = 15
    

    def process_noise(self):
        dt = self.dt
        
        q_acc = self.q_a_psd * dt
        q_vel = q_acc * dt * 0.5
        q_pos = q_vel * dt * 2/3.0
        q_w   = self.q_w_psd * dt

        return np.diag([q_pos, q_pos, q_pos,
                        q_vel, q_vel, q_vel,
                        q_w,   q_w,   q_w,
                        0.0,   0.0,   0.0,
                        0.0,   0.0,   0.0])

    def F(self):
        x = self.x
        T_body_to_inrtl = np.identity(3)
        
        F = np.zeros((self.N, self.N))
        F[9:self.N,9:self.N] = np.diag(-self.beta)
        
        F[0:3,3:6] = np.identity(3)
        F[3:6,0:3] = G(x.eci[0:3], x.mu_earth) + G(x.lci[0:3], x.mu_moon)
        F[3:6,6:9] = -pq.skew(x.a_meas_inrtl).dot(T_body_to_inrtl)
        F[3:6,9:12] = -T_body_to_inrtl

        F[6:9,6:9] = -pq.skew(x.w_meas_inrtl)
        F[6:9,12:15] = -np.identity(3)

        return F
    
    def state_transition(self):
        F = self.F()

        Phi = np.identity(self.N) + F * self.dt

        return Phi

    def att_update(self, x, P, plot=False):
        # measurement covariance:
        # FIXME: Needs to be adjusted for star tracker choice
        R_att = np.diag(x.params.att_sigma ** 2) # arcseconds to radians
        

        if plot:
            from plot_lincov import plot_covariance
            import matplotlib.pyplot as plt
            plot_covariance(R, xlabel='x (rad)', ylabel='y (rad)', zlabel='z (rad')
            plt.show()
        
        # measurement linearization / measurement sensitivity matrix (a Jacobian)
        H = np.zeros((3, self.N))
        H[0:3,6:9] = x.T_body_to_att
        
        return H, R_att

    def altimeter_update(self, x, P, rel):
        if rel in ('earth', 399):
            body_id = 399
            x_pci   = x.eci
        elif rel in ('moon', 301):
            body_id = 301
            x_pci   = x.lci

        u_alt = np.array([0.0, 0.0, 1.0]) # Boresight is +z-axis

        R_alt = np.array([[x.params.altimeter_sigma]])
        H = np.zeros((1, self.N))

        # If we cared about the range-rate component, we would want to do
        # this in an PCPF frame, but let's just skip that and do PCI 
        u_pci = x.T_inrtl_to_body.T.dot(self.T_alt_to_body.dot( u_alt ))

        # Re-center at the spacecraft
        r_rel_pci = x_pci[0:3] + u_pci
        H[0,0:3] = r_rel_pci / norm(r_rel_pci)

        return H, R_alt


    def velocimeter_update(self, x, P, rel):
        if rel in ('earth', 399):
            body_id = 399
            x_pci   = x.eci
            Tdot_inrtl_to_pcpf = self.T_inrtl_to_ecef[3:6,0:3]
        elif rel in ('moon', 301):
            body_id = 301
            x_pci   = x.lci
            Tdot_inrtl_to_pcpf = self.T_inrtl_to_lclf[3:6,0:3]

        # These unit vectors are computed from azimuths and elevations
        # given in the yml. The computation is done in yaml_loader.py.
        # They are in the NDL frame.
        T_inrtl_to_vel = x.T_inrtl_to_body[0:3,0:3].T.dot(self.T_vel_to_body).T
        T_inrtl_to_vel_a = T_inrtl_to_vel.dot(x.params.T_velocimeter_a)
        T_inrtl_to_vel_b = T_inrtl_to_vel.dot(x.params.T_velocimeter_b)
        T_inrtl_to_vel_c = T_inrtl_to_vel.dot(x.params.T_velocimeter_c)

        zhat = np.array([0.0, 0.0, 1.0])
        
        u_a_inrtl = zhat.dot(T_inrtl_to_vel_a)
        u_b_inrtl = zhat.dot(T_inrtl_to_vel_b)
        u_c_inrtl = zhat.dot(T_inrtl_to_vel_c)

        # Get distance from laser to nearest point on planet surface. If zero, we can use the measurement
        dist_a = spice.npedln(a,b,c, x_pci[0:3], u_a_inrtl[0,:])[1]
        dist_b = spice.npedln(a,b,c, x_pci[0:3], u_b_inrtl[1,:])[1]
        dist_c = spice.npedln(a,b,c, x_pci[0:3], u_c_inrtl[2,:])[1]

        H = np.zeros((3, self.N))
        if dist_a == 0:
            H[0,0:3] = u_a_inrtl.dot(Tdot_inrtl_to_pcpf.T) # this second term is negative omega cross, hopefully
            H[0,3:6] = u_a_inrtl
        if dist_b == 0:
            H[1,0:3] = u_b_inrtl.dot(Tdot_inrtl_to_pcpf.T) # this second term is negative omega cross, hopefully
            H[1,3:6] = u_b_inrtl
        if dist_c == 0:
            H[2,0:3] = u_c_inrtl.dot(Tdot_inrtl_to_pcpf.T) # this second term is negative omega cross, hopefully
            H[2,3:6] = u_c_inrtl

        R_vel = np.eye(3) * x.params.velocimeter_sigma**2

        return H, R_vel
        

    def horizon_update(self, x, P, rel, plot=False):
        if rel in ('earth', 399):
            body_id = 399
            r_pci   = x.eci[0:3]
        elif rel in ('moon', 301):
            body_id = 301
            r_pci   = x.lci[0:3]
            
        R_cam = horizon.covariance(x.time, body_id,
                                   fpa_size  = x.params.horizon_fpa_size,
                                   fov       = x.params.horizon_fov,
                                   theta_max = x.params.horizon_theta_max,
                                   sigma_pix = x.params.horizon_sigma_pix,
                                   n_max     = x.params.horizon_n_max )
        

        if plot:
            from plot_lincov import plot_covariance
            import matplotlib.pyplot as plt
            T_lvlh = frames.compute_T_inrtl_to_lvlh(x.lci)[0:3,0:3]
            plot_covariance(T_lvlh.dot(R).dot(T_lvlh.T), xlabel='downtrack (m)', ylabel='crosstrack (m)', zlabel='radial (m)')
            plt.show()
            
        H = np.zeros((3, self.N))
        H[0:3,0:3] = x.T_body_to_horizon_cam.dot(x.T_inrtl_to_body)
        H[0:3,6:9] = x.T_body_to_horizon_cam.dot(pq.skew(x.T_inrtl_to_body.dot(r_pci)))

        return H, R_cam

    def twoway_doppler_update(self, x, P):
        H = np.zeros((len(x.visible_from), self.N))
        R = np.zeros((len(x.visible_from), len(x.visible_from)))
        for ii, name in enumerate(x.visible_from):
            ground_id = x.params.ground_stations[name]
            # Need times so we can get positions of ground stations slightly earlier
            t1, tau12 = spice.ltime(x.time, x.loader.object_id, "<-", ground_id)
            t3, tau23 = spice.ltime(x.time, x.loader.object_id, "->", ground_id)

            T_ecef_to_inrtl1 = spice.sxform('ITRF93', 'J2000', t1)
            T_ecef_to_inrtl3 = spice.sxform('ITRF93', 'J2000', t3)

            x_station_ecef  = np.hstack(( State.r_station_ecef[name], np.zeros(3) ))
            x1 = T_ecef_to_inrtl1.dot(x_station_ecef)
            x2 = x.eci
            x3 = T_ecef_to_inrtl3.dot(x_station_ecef)
            r1 = x1[0:3]
            v1 = x1[3:6]
            r2 = x2[0:3]
            v2 = x2[3:6]
            r3 = x3[0:3]
            v3 = x3[3:6]
            r23 = norm(r3 - r2)
            r12 = norm(r2 - r1)

            dG1 = -(r2 - r1) * (r2 - r1).dot(v2 - v1) / r12**3
            dG2 =  (v2 - v1) / r12
            dG3 =  (r3 - r2) * (r3 - r2).dot(v3 - v2) / r23**3
            dG4 = -(v3 - v2) / r23
            dF_dr = dG1 + dG2 + dG3 + dG4
            dF_dv = (r2 - r1) / r12 - (r3 - r2) / r23

            # Note that we drop the constant frequency term so we can
            # state our sigma in distance and distance/time. We also
            # normalize here so sigma is also normalized.
            H[ii,0:3] = dF_dr * 0.5
            H[ii,3:6] = dF_dv * 0.5

            R[ii,ii] = x.params.twoway_doppler_sigma**2

        return H, R

    def twoway_range_update(self, x, P):
        H = np.zeros((len(x.visible_from), self.N))
        R = np.zeros((len(x.visible_from), len(x.visible_from)))
        for ii, name in enumerate(x.visible_from):
            ground_id = x.params.ground_stations[name]

            # Get locations of ground stations at t1 and t3
            t1, tau12 = spice.ltime(x.time, x.loader.object_id, "<-", ground_id)
            t3, tau23 = spice.ltime(x.time, x.loader.object_id, "->", ground_id)

            T_ecef_to_inrtl1 = spice.sxform('ITRF93', 'J2000', t1)
            T_ecef_to_inrtl3 = spice.sxform('ITRF93', 'J2000', t3)

            x_station_ecef  = np.hstack(( State.r_station_ecef[name], np.zeros(3) ))
            r1 = T_ecef_to_inrtl1.dot(x_station_ecef)[0:3]
            r2 = x.eci[0:3]
            r3 = T_ecef_to_inrtl3.dot(x_station_ecef)[0:3]
            r12 = norm(r2 - r1)
            r23 = norm(r3 - r2)

            # We drop the constant frequency term and normalize here
            # so that sigma is given in meters and is also normalized.
            H[ii,0:3] = ((r2 - r1) / r12 - (r3 - r2) / r23) * 0.5
            R[ii,ii]  = x.params.twoway_range_sigma**2

        return H, R


    def update(self, meas_type):
        """Attempt to process a measurement update for some measurement type"""
        updated = False

        time = self.time
        x    = self.x
        P    = self.P
        R    = None
        
        if time > self.meas_last[meas_type] + self.meas_dt[meas_type]:
            
            if meas_type == 'att':
                H, R = self.att_update(x, P)
                updated = True
            elif meas_type == 'horizon_moon':
                if x.horizon_moon_enabled:
                    H, R = self.horizon_update(x, P, 'moon')
                    updated = True
            elif meas_type == 'horizon_earth':
                if x.horizon_earth_enabled:
                    H, R = self.horizon_update(x, P, 'earth')
                    updated = True
            elif meas_type == 'twoway_range':
                if len(self.x.visible_from) > 0:
                    H, R = self.twoway_range_update(x, P)
                    updated = True
            elif meas_type == 'twoway_doppler':
                if len(self.x.visible_from) > 0:
                    H, R = self.twoway_doppler_update(x, P)
                    updated = True
            elif meas_type == 'altimeter':
                if self.alt_altimeter_earth >= self.params.altimeter_min_alt and self.alt_altimeter_earth <= self.params.altimeter_max_alt:
                    H, R = self.altimeter_update(x, P, 'earth')
                    updated = True
                elif self.alt_altimeter_moon >= self.params.altimeter_min_alt and self.alt_altimeter_moon <= self.params.altimeter_max_alt:
                    H, R = self.altimeter_update(x, P, 'moon')
                    updated = True
            elif meas_type == 'velocimeter':
                if self.alt_velocimeter_earth >= self.params.velocimeter_min_alt and self.alt_velocimeter_earth <= self.params.velocimeter_max_alt:
                    H, R = self.velocimeter_update(x, P, 'earth')
                    updated = True
                elif self.alt_velocimeter_moon >= self.params.velocimeter_min_alt and self.alt_velocimeter_moon <= self.params.velocimeter_max_alt:
                    H, R = self.velocimeter_update(x, P, 'moon')
                    updated = True
            else:
                raise NotImplemented("unrecognized update type '{}'".format(meas_type))

            if updated:
                self.meas_last[meas_type] = time
                PHt = P.dot(H.T)
                
                if len(H.shape) == 1 or H.shape[0] == 1: # perform a scalar update
                    W = H.dot(PHt) + R
                    K = PHt / W[0,0]

                    # Scalar joseph update
                    self.P = P - K.dot(H.dot(P)) - PHt.dot(K.T) + (K*W).dot(K.T)

                else: # perform a vector update
                    K = PHt.dot(inv(H.dot(PHt) + R))

                    # Vector Joseph update
                    I_minus_KH = np.identity(K.shape[0]) - K.dot(H)
                    self.P     = I_minus_KH.dot(P).dot(I_minus_KH.T) + K.dot(R).dot(K.T)

        return updated, R

    def propagate(self):
        """Propagate covariance matrix forward in time by dt"""
        self.x = State(self.time, loader = self.loader, params = self.params)
        Phi = self.Phi = self.state_transition()
        P = self.P
        Q = self.Q
        self.P = Phi.dot(P.dot(Phi.T)) + Q
        

    def save_data(self, name, time, cols, resume = False):
        d = {'time': time}
        for key in cols:
            if len(cols[key].shape) == 1:
                d[key] = cols[key]
            elif len(cols[key].shape) == 3:
                d[key + 'xx'] = cols[key][0,0,:]
                d[key + 'xy'] = cols[key][0,1,:]
                d[key + 'xz'] = cols[key][0,2,:]
                d[key + 'yy'] = cols[key][1,1,:]
                d[key + 'yz'] = cols[key][1,2,:]
                d[key + 'zz'] = cols[key][2,2,:]
            elif cols[key].shape[1] == 3:
                d[key + 'x'] = cols[key][:,0]
                d[key + 'y'] = cols[key][:,1]
                d[key + 'z'] = cols[key][:,2]
            elif cols[key].shape[1] == 4:
                d[key + 'w'] = cols[key][:,0]
                d[key + 'x'] = cols[key][:,1]
                d[key + 'y'] = cols[key][:,2]
                d[key + 'z'] = cols[key][:,3]
                
        frame = pd.DataFrame(d)

        # First make sure directory exists
        pathlib.Path("output/{}".format(self.label)).mkdir(parents=True, exist_ok=True)
        
        filename = "output/{}/{}.{:04d}.feather".format(self.label, name, self.count)
        frame.to_feather(filename)

    def save_metadata(self, snapshot_label = None):
        metadata = {
            'meas_last': self.meas_last,
            'start':     self.start,
            'meas_dt':   self.meas_dt,
            'count':     self.count,
            'dt':        self.dt,
            'order':     self.order
            }

        filename = LinCov.metadata_filename(self.label, self.count, snapshot_label)
        pickle.dump( metadata, open(filename, 'wb') )

    @classmethod
    def metadata_filename(self, label, count = None, snapshot_label = None):
        if snapshot_label:
            filename = "output/{}/metadata.{}.p".format(label, snapshot_label)
        else:
            filename = "output/{}/metadata.{:04d}.p".format(label, count)
            
        return filename
        
    @classmethod
    def load_metadata(self, label, count, snapshot_label = None):
        filename = LinCov.metadata_filename(label, count, snapshot_label)            
        try:
            metadata = pickle.load( open(filename, 'rb') )
            return metadata
        except IOError:
            return None


    @classmethod
    def save_covariance(self, label, P, time, count = 0, snapshot_label = None):
        if snapshot_label:
            suffix = "{}.npy".format(snapshot_label)
        else:
            suffix = "{:04d}.npy".format(count)
            
        with open("output/{}/P.{}".format(label, suffix), 'wb') as P_file:
            np.save(P_file, P)
        with open("output/{}/time.{}".format(label, suffix), 'wb') as time_file:
            np.save(time_file, time)


    @classmethod
    def find_latest_count(self, label):
        """Find the latest iteration of the analysis that has been run for a
        given run name.

        Returns:
            The count in the time filename or None if no files present.
        """
        import os
        import glob

        old_path = os.getcwd()

        new_path = "output/{}".format(label)
        if os.path.isdir(new_path):
            os.chdir(new_path)
        else:
            return None

        files = glob.glob("time.*.npy")
        counts = []
        for filename in files:
            try:
                counts.append( int(filename.split('.')[1]) )
            except ValueError:
                continue

        os.chdir(old_path)

        if len(counts) == 0:
            return None
        
        return sorted(counts)[-1]
            
    @classmethod
    def load_covariance(self, label, count = 0, snapshot_label = None):
        if snapshot_label:
            suffix = "{}.npy".format(snapshot_label)
        else:
            suffix = "{:04d}.npy".format(count)
        
        with open("output/{}/P.{}".format(label, suffix), 'rb') as P_file:
            P = np.load(P_file)
        with open("output/{}/time.{}".format(label, suffix), 'rb') as time_file:
            time = float(np.load(time_file))
            
        return P, time


    @classmethod
    def create_label(self, loader,
                     label         = 'f9',
                     sample_method = sample_f9_gto_covariance):
        """Create a new LinCov run label, generating a new covariance.

        Args:
          loader         SpiceLoader object
          label          name for the covariance (default is f9)
          sample_method  method to call for generating covariance (default
                         is sample_f9_gto_covariance)

        Returns:
          
        """
        
        x = State(self.start, loader = loader)
        time = x.time
        
        P = sample_method(x)
        LinCov.save_covariance(label, P, time)
        

    @classmethod
    def start_from(self, loader, label,
                   count          = None,
                   copy_from      = None,
                   snapshot_label = None):
        """Resume from a previous LinCov by loading metadata, time, and
        covariance.
        
        Args:
          loader          SpiceLoader object
          label           specifies the label for the run
          count           optional run number (it will pick the most recent if
                          you don't specify one and don't provide a snapshot label)
          copy_from       start by copying data from a different run
          snapshot_label  load a snapshot instead of a count
          
        Returns:
            A LinCov object, fully instantiated.

        """

        if copy_from is None:
            copy_from = label
        
        # If no index is given, see if we can reconstruct it.
        if snapshot_label is None and count is None:
            count = LinCov.find_latest_count(copy_from)

        # Try to load metadata from the other run. If that doesn't
        # work, see if there's a yml configuration for the desired
        # label.
        metadata = LinCov.load_metadata(copy_from, count, snapshot_label = snapshot_label)
        config   = YamlLoader(label)
        block_dt = config.block_dt
        if metadata is None:
            metadata = config.as_metadata()
                
        P, time = LinCov.load_covariance(copy_from, count = count, snapshot_label = snapshot_label)

        # If we were given a snapshot label, we need to determine
        # which count it falls in the middle of.
        if snapshot_label and count is None:
            count      = loader.find_count(time, config.block_dt)
            block_dt   = (count + 2) * config.block_dt + loader.start - time
            if count < 0: count = 0
        
        return LinCov(loader, label, count + 1, P, time,
                      metadata['dt'],
                      metadata['meas_dt'],
                      metadata['meas_last'],
                      metadata['order'],
                      block_dt,
                      config.params)

    
    def __init__(self, loader, label, count, P, time, dt, meas_dt, meas_last, order, block_dt, params):
        """Constructor, which is basically only called internally."""
        self.loader    = loader
        self.label     = label
        self.count     = count
        self.P         = P
        self.time      = time
        self.dt        = dt
        self.meas_dt   = meas_dt
        self.meas_last = meas_last
        self.order     = order
        self.block_dt  = block_dt
        self.end       = time
        self.finished  = False
        self.params    = params

        # Set process noise parameters
        self.tau              = np.array(params['tau'])
        self.beta             = 1/self.tau
        self.q_a_psd          = max(params.q_a_psd_imu, params.q_a_psd_dynamics)
        self.q_w_psd          = params.q_w_psd

        # Generate process noise matrix
        self.Q = self.process_noise()

        # Setup end times
        self.prepare_next()


    def prepare_next(self):
        """Get ready for next run."""
        self.start = self.end
        end_time   = self.loader.end

        if end_time > self.start + self.block_dt:
            self.end = self.start + self.block_dt
        else:
            self.end = end_time

    def run(self):
        """This is the actual loop that runs the linear covariance
        analysis."""
        dt       = self.dt
        count    = self.count
        loader   = self.loader
        
        times    = []
        
        sr       = [] # position sigma
        sv       = [] # velocity sigma
        satt     = []
        sba      = []
        sbg      = []
        elvlh_sr = []
        elvlh_sv = []
        llvlh_sr = []
        llvlh_sv = []


        update_times = {}
        update_Rs    = {}
        for meas_type in self.meas_dt:
            update_times[meas_type] = []
            update_Rs[meas_type] = []

        environment_times = []
        earth_angle       = []
        moon_angle        = []
        earth_phase_angle = []
        moon_phase_angle  = []
        range_earth       = []
        range_moon        = []
        station_elevations = {}
        for station in self.params.ground_stations:
            station_elevations[station] = []

        self.time += dt

        # Loop until completion
        while self.time < self.end:
            self.propagate()
            
            P = self.P

            # These don't change pre- and post-update in a LinCov (but
            # they do in a Kalman filter)
            T_inrtl_to_elvlh = compute_T_inrtl_to_lvlh(self.x.eci)
            T_inrtl_to_llvlh = compute_T_inrtl_to_lvlh(self.x.lci)
            
            # Pre-update logging
            times.append( self.time )
            sr.append(np.sqrt(np.diag(P[0:3,0:3])))
            sv.append(np.sqrt(np.diag(P[3:6,3:6])))
            satt.append(np.sqrt(np.diag(P[6:9,6:9])))
            sba.append(np.sqrt(np.diag(P[9:12,9:12])))
            sbg.append(np.sqrt(np.diag(P[12:15,12:15])))

            P_elvlh = T_inrtl_to_elvlh.dot(P[0:6,0:6]).dot(T_inrtl_to_elvlh.T)
            P_llvlh = T_inrtl_to_llvlh.dot(P[0:6,0:6]).dot(T_inrtl_to_llvlh.T)
            
            elvlh_sr.append(np.sqrt(np.diag(P_elvlh[0:3,0:3])))
            elvlh_sv.append(np.sqrt(np.diag(P_elvlh[3:6,3:6])))

            llvlh_sr.append(np.sqrt(np.diag(P_llvlh[0:3,0:3])))
            llvlh_sv.append(np.sqrt(np.diag(P_llvlh[3:6,3:6])))

            environment_times.append( self.time )
            earth_angle.append( self.x.earth_angle )
            moon_angle.append( self.x.moon_angle )
            earth_phase_angle.append( self.x.earth_phase_angle )
            moon_phase_angle.append( self.x.moon_phase_angle )
            range_earth.append( self.x.range_earth )
            range_moon.append( self.x.range_moon )
            
            for station in self.params.ground_stations:
                station_elevations[station].append( self.x.elevation_from[station] )

            yield self, 'propagate'

            updated = False
            for meas_type in self.order:
                if self.time >= self.meas_last[meas_type] + self.meas_dt[meas_type]:
                    #print("{}: updating {}".format(time, meas_type))
                    updated, R = self.update(meas_type)

                    if updated:
                        update_times[meas_type].append( self.time )
                        update_Rs[meas_type].append( R )

                    yield self, meas_type

            

            # Post-update logging
            P = self.P
            times.append(self.time)
            sr.append(np.sqrt(np.diag(P[0:3,0:3])))
            sv.append(np.sqrt(np.diag(P[3:6,3:6])))
            satt.append(np.sqrt(np.diag(P[6:9,6:9])))
            sba.append(np.sqrt(np.diag(P[9:12,9:12])))
            sbg.append(np.sqrt(np.diag(P[12:15,12:15])))

            P_elvlh = T_inrtl_to_elvlh.dot(P[0:6,0:6]).dot(T_inrtl_to_elvlh.T)
            P_llvlh = T_inrtl_to_llvlh.dot(P[0:6,0:6]).dot(T_inrtl_to_llvlh.T)
            elvlh_sr.append(np.sqrt(np.diag(P_elvlh[0:3,0:3])))
            elvlh_sv.append(np.sqrt(np.diag(P_elvlh[3:6,3:6])))
            
            llvlh_sr.append(np.sqrt(np.diag(P_llvlh[0:3,0:3])))
            llvlh_sv.append(np.sqrt(np.diag(P_llvlh[3:6,3:6])))
            
            
            self.time += self.dt

        if self.time > self.end: # Make sure we don't go past the requested end
            self.time = self.end
        if self.time >= self.loader.end:
            self.finished = True
        else:
            self.finished = False

        # Save state covariances
        self.save_data('state_sigma', np.hstack(times), {
            'sr': np.vstack(sr),
            'sv': np.vstack(sv),
            'satt': np.vstack(satt),
            'sba': np.vstack(sba),
            'sbg': np.vstack(sbg),
            'elvlh_sr': np.vstack(elvlh_sr),
            'elvlh_sv': np.vstack(elvlh_sv),
            'llvlh_sr': np.vstack(llvlh_sr),
            'llvlh_sv': np.vstack(llvlh_sv)
        })

        # Save extra state information
        env_dict = {
            'earth_angle': np.hstack(earth_angle).astype(np.float32),
            'moon_angle': np.hstack(moon_angle).astype(np.float32),
            'earth_phase_angle': np.hstack(earth_phase_angle).astype(np.float32),
            'moon_phase_angle': np.hstack(moon_phase_angle).astype(np.float32),
            'range_earth': np.hstack(range_earth),
            'range_moon': np.hstack(range_moon)
        }
        # Add ground station elevations
        for station in station_elevations:
            env_dict['elevation_' + station] = np.hstack(station_elevations[station]).astype(np.float32)
            
        self.save_data('environment', np.hstack(environment_times), env_dict)

        # Save measurement covariances
        for meas_type in update_times:
            if meas_type in ('horizon_earth', 'horizon_moon'):
                if len(update_times[meas_type]) > 0:
                    self.save_data(meas_type,
                                   np.hstack(update_times[meas_type]), {
                                       'R': np.dstack(update_Rs[meas_type])
                                   })

        # Save information we need for resuming
        self.save_metadata()
        LinCov.save_covariance(self.label, self.P, self.time, self.count)

        self.count += 1
        self.prepare_next()
