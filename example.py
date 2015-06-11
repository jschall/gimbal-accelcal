import setup_accelcal
import numpy as np
from math import *

GRAVITY_MSS = 9.80655

def get_samples(bias=np.matrix('0. ; 0. ; 0.'), mat=np.matrix('1. 0. 0. ; 0. 1. 0. ; 0. 0. 1.')):
    samples = [
    mat.I*np.matrix([[GRAVITY_MSS, 0, 0]]).T+bias,
    mat.I*np.matrix([[-GRAVITY_MSS, 0, 0]]).T+bias,
    mat.I*np.matrix([[0, GRAVITY_MSS, 0]]).T+bias,
    mat.I*np.matrix([[0, -GRAVITY_MSS, 0]]).T+bias,
    mat.I*np.matrix([[0, 0, GRAVITY_MSS]]).T+bias,
    mat.I*np.matrix([[0, 0, -GRAVITY_MSS]]).T+bias
    ]

    for i in range(len(samples)):
        samples[i] = np.squeeze(np.asarray(samples[i]))

    return samples

s = get_samples(np.matrix('1. ; 0. ; 0.'), np.matrix('1.1 0. 0. ; 0. .9 0. ; 0. 0. .97'))
p = setup_accelcal.calibrate_accel_6dof(s)
e = setup_accelcal.calc_level_euler_rpy(p, s[5])
print p
print e*degrees(1)
