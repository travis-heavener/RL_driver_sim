from math import sin, pi
import numpy as np
from scipy.interpolate import splprep, splev

import consts

# #############################################
#
# Point class
#
# #############################################

class Point:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def astuple(self):
        return (self.x, self.y)

    def __add__(self, p2):
        return Point(self.x + p2.x, self.y + p2.y)
    
    def __sub__(self, p2):
        return Point(self.x - p2.x, self.y - p2.y)
    
    def __mul__(self, s):
        return Point(self.x * s, self.y * s)
    
    def __rmul__(self, s):
        return Point(self.x * s, self.y * s)
    
    def cross2d(self, p2):
        return self.x * p2.y - self.y * p2.x

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

#
# check if a line segment intersects with another segment
#
def __check_orientation(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

def do_segments_intersect(s1, s2) -> bool:
    # ref #1: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    # ref #2: https://stackoverflow.com/a/9997374
    # check orientation of lines against each other
    p1, q1, p2, q2 = *s1, *s2
    A, B = Point(*p1), Point(*q1)
    C, D = Point(*p2), Point(*q2)

    return (__check_orientation(A, C, D) != __check_orientation(B, C, D) and
            __check_orientation(A, B, C) != __check_orientation(A, B, D))

# return the intersection point of two segments
# ref: https://stackoverflow.com/a/565282
# all those multi lectures and linear algebra lessons to look this up... smh
def get_segment_intersection(s1, s2) -> np.ndarray:
    p1, q1, p2, q2 = *s1, *s2
    A, B = Point(*p1), Point(*q1)
    C, D = Point(*p2), Point(*q2)
    
    # r & s are the vectors aligned with the direction of each segment
    # (A, B) is (A, A+r) -> B=A+r
    r, s = B-A, D-C

    # segments intersect at A + tr = B + us
    r_cross_x = r.cross2d(s)
    if r_cross_x == 0: return None # no intersection, prevent div by 0

    t = (C - A).cross2d(s) / r_cross_x

    return np.array( (A + t*r).astuple() )