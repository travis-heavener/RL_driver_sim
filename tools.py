from math import sin, pi
import numpy as np
from scipy.interpolate import splprep, splev

import consts

# #############################################
#
#               vehicle tools
#
# #############################################

# calculate the engine torque and horsepower at the given rpms
__HALF_MAX_TORQUE = 0.5 * consts.MAX_TORQUE
__VEHICLE_MASS = consts.VEHICLE_WEIGHT / 2.205 # in kg

def getEngineTQ(rpms: int, throttle: float) -> float:
    if rpms < consts.IDLE_RPMS:
        return 0
    return throttle * __HALF_MAX_TORQUE * (sin((rpms / 3000) - 0.2) + 1)

def getEngineHP(rpms: int, throttle: float) -> float:
    return getEngineTQ(rpms, throttle) * rpms / 5252

# calculate the resulting net force the vehicle experiences at the given moment
def calcVehicleForce(gear: int, rpms: int, throttle: float) -> float:
    speed_m_s = getVehicleMPH(gear, rpms) / 2.237 # convert mph to m/s
    f_engine = getEngineHP(rpms, throttle) * 745.7 / speed_m_s # 1 HP ~ 745.7 Watts
    f_drag = -consts.DRAG_COEF * (speed_m_s ** 2)
    f_rolling = -consts.ROLLING_FRICTION * __VEHICLE_MASS * consts.GRAVITY_ACCEL
    t_ebrake = -consts.ENGINE_BRAKE_COEF * rpms
    f_ebrake = t_ebrake * consts.GEAR_RATIOS[gear-1] * consts.FINAL_DRIVE / (consts.ROLLING_DIAMETER * 0.0254) # in to meters
    return f_engine + f_drag + f_rolling + f_ebrake

# calculate speed of car at a given moment
def getVehicleMPH(gear: int, rpms: int) -> float:
    wheel_rpms = rpms / (consts.FINAL_DRIVE * consts.GEAR_RATIOS[gear-1])
    return wheel_rpms * consts.ROLLING_DIAMETER * pi / 1056 # simplified conversion from m/s to mph

# #############################################
#
#               track tools
#
# #############################################

# create inner & outer track borders from central spline curve
def __gen_offset_line(tck, u_final, line, distance):
    # determine tangent vector at point on spline
    dx, dy = splev(u_final, tck, der=1)
    tangent_mag = np.sqrt(dx**2 + dy**2)
    
    # normalize tangent vector
    dx /= tangent_mag
    dy /= tangent_mag
    
    # determine the boundary offset from the central spline curve
    return line + distance * np.stack((-dy, dx), axis=-1)

# generate spline curve from track inflection vertices
# ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html#scipy.interpolate.splprep
def gen_outer_spline(vertices):
    tck = splprep(vertices.T, s=0, k=3, per=True)[0]
    u_final = np.linspace(0, 1, num=200)
    center_spline = np.array(splev(u_final, tck))
    width_px = consts.TRACK_WIDTH_Y * consts.PX_YARD_RATIO
    return tck, center_spline.T

def gen_inner_spline(tck, outer_spline):
    u_final = np.linspace(0, 1, num=200)
    width_px = consts.TRACK_WIDTH_Y * consts.PX_YARD_RATIO
    return __gen_offset_line(tck, u_final, outer_spline, -width_px)