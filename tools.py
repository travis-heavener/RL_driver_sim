import math
import numpy as np
import pygame
import pygame.gfxdraw
from scipy.interpolate import splprep, splev
from tensorflow import keras
from keras import layers, models, losses, optimizers, initializers
from numba import njit
import os
import types

import consts

@njit
def log(*args): print("Note:", *args)
@njit
def warn(*args): print("Warn:", *args)

# #############################################
#
#               vehicle tools
#
# #############################################

# calculate the engine torque and horsepower at the given rpms
__HALF_MAX_TORQUE = 0.5 * consts.MAX_TORQUE
__VEHICLE_MASS = consts.VEHICLE_WEIGHT / 2.205 # in kg

# employ custom torque function
@njit
def get_engine_TQ(rpms: int, throttle: float) -> float:
    if rpms < consts.IDLE_RPMS:
        return 0
    return throttle * __HALF_MAX_TORQUE * (math.sin((rpms / 3000) - 0.2) + 1)

@njit
def get_engine_HP(rpms: int, throttle: float) -> float:
    return get_engine_TQ(rpms, throttle) * rpms / 5252

# calculate the resulting net force the vehicle experiences at the given moment
@njit
def calcVehicleForce(gear: int, rpms: int, throttle: float) -> float:
    # handle speeding up and braking
    speed = rpms_to_speed(gear, max(consts.IDLE_RPMS, rpms))
    f_engine = 0
    if throttle > 0:
        f_engine = get_engine_HP(rpms, throttle) * 745.7 / speed # 1 HP ~ 745.7 Watts
    else:
        f_engine = -consts.BRAKING_FRICTION * -throttle * __VEHICLE_MASS * consts.GRAVITY_ACCEL

    f_drag = -consts.DRAG_COEF * (speed ** 2)

    f_rolling = -consts.ROLLING_FRICTION * __VEHICLE_MASS * consts.GRAVITY_ACCEL

    t_ebrake = -consts.ENGINE_BRAKE_COEF * rpms
    f_ebrake = t_ebrake * consts.GEAR_RATIOS[gear-1] * consts.FINAL_DRIVE / consts.ROLLING_DIAMETER_M
    return f_engine + f_drag + f_rolling + f_ebrake

# calculate speed of car at a given moment
@njit
def rpms_to_speed(gear: int, rpms: int) -> float:
    wheel_rpms = rpms / (consts.FINAL_DRIVE * consts.GEAR_RATIOS[gear-1])
    circumference = consts.ROLLING_DIAMETER_M * math.pi # meters
    return wheel_rpms * circumference / 60 # from meters/minute to meters/second

# return the necessary RPMs for the given speed in the given gear
@njit
def speed_to_rpms(speed: float, gear: int) -> float:
    circumference = consts.ROLLING_DIAMETER_M * math.pi # meters
    ratio = consts.FINAL_DRIVE * consts.GEAR_RATIOS[gear-1]
    return 60 * speed * ratio / circumference

# convert mph to m/s
@njit
def mph2ms(speed: float) -> float:
    return speed / 2.237

# convert m/s to mph
@njit
def ms2mph(speed: float) -> float: return speed * 2.237

# convert lbs to kg
@njit
def lbs2kg(lbs: float) -> float: return lbs / 2.205

########## raycasting ##########

# returns all the segments that make up the walls of the track and driver bboxes
def get_obstacle_segs(track_poly: np.ndarray, bboxes: list[np.ndarray]) -> np.ndarray:
    segs = []

    # track segments
    len_track_poly = len(track_poly)
    for i in range(len_track_poly):
        # skip implicit segments (that connect outer to inner walls & vice versa)
        # middle 2 pts connect outer to inner, outer 2 pts connect inner to outer
        if i == len_track_poly-1 or i+1 == len_track_poly // 2: continue

        segs.append( np.array([track_poly[i], track_poly[(i+1) % len_track_poly]]) )

    # driver bbox segments
    for bbox in bboxes:
        len_bbox = len(bbox)
        for i in range(len_bbox):
            segs.append( np.array([bbox[i], bbox[(i+1) % len_bbox]]) )

    return np.array(segs)

#
# searches for the nearest obstacle from the given angle out
# returns the distance to the nearest obstacle
#
@njit
def emit_ray(obstacle_segs: np.ndarray, ray: np.ndarray, return_on_hit=False) -> float:
    closest_obstacle = consts.SENSOR_NOT_FOUND

    # check for intersections with obstacle hitbox segments
    for seg in obstacle_segs:
        # cast ray
        if do_segments_intersect(ray, seg):
            if return_on_hit: # return a value less than SENSOR_MAX_RANGE
                return -1
        
            intersection = get_segment_intersection(ray, seg)
            if intersection is None: continue

            components = intersection - ray[0]
            dist_m = math.hypot(components[0], components[1]) / consts.PX_METER_RATIO

            if dist_m < closest_obstacle: closest_obstacle = dist_m

    return closest_obstacle

#
# Returns a tuple of the lower and upper shift points for the current gear,
# factoring in driver mood.
#
@njit
def get_shift_points(gear: int, rev_bias: float) -> tuple[int, int]:
    lower = None if gear == 1 else consts.LOWER_SHIFT_POINTS[gear-2]
    if lower is not None:
        lower -= 0.67 * (1-rev_bias) * (consts.MAX_RPMS - consts.REDLINE_RPMS)
        lower = max(lower, consts.IDLE_RPMS)
    upper = consts.MAX_RPMS - 2 * (1-rev_bias) * (consts.MAX_RPMS - consts.REDLINE_RPMS)
    return (lower, upper)

# get the lowest (target) gear at a given wheel speed
@njit
def get_target_gear(speed: float, rev_bias: float) -> float:
    num_gears = len(consts.GEAR_RATIOS)
    for gear in range(1, num_gears+1):
        shift_pts = get_shift_points(gear, rev_bias) # get shift points for gear
        rpms = speed_to_rpms(speed, gear) # get rpms in gear

        if rpms < shift_pts[1]: # rpms are low enough to fit in the gear
            return gear
    
    # base case, last gear
    return num_gears

# returns True when a downshift would result in a money shift
@njit
def is_moneyshift_possible(speed: float, gear: int, rev_bias: float) -> bool:
    if gear <= 1: return True
    return speed_to_rpms(speed, gear-1) > consts.MAX_RPMS

# returns True when an upshift would result in lugging
@njit
def is_lugging_possible(speed: float, gear: int, rev_bias: float) -> bool:
    if gear >= len(consts.GEAR_RATIOS): return True
    return speed_to_rpms(speed, gear+1) < get_shift_points(gear+1, rev_bias)[0]

# #############################################
#
#               track tools
#
# #############################################

# draw antialiased lines (ref: https://stackoverflow.com/a/30599392)
def draw_aa_lines(window, lines, color, width):
    num_lines = len(lines)
    for i in range(num_lines):
        p0 = lines[i]
        p1 = lines[(i+1) % num_lines]
        draw_aa_line(window, np.array((p0, p1)), color, width)

def draw_aa_line(window, line, color, width):
    p0, p1 = line

    # generate polygon from line
    mid = (p0 + p1) / 2
    length = math.hypot(*(p1 - p0))
    theta = math.atan2(p0[1] - p1[1], p0[0] - p1[0])
    cos, sin = math.cos(theta), math.sin(theta)
    len_cos, len_sin = length * cos / 2, length * sin / 2
    wid_cos, wid_sin = width * cos / 2, width * sin / 2

    # calculate vertices
    top_left  = (mid[0] + len_cos - wid_sin, mid[1] + wid_cos + len_sin)
    top_right = (mid[0] - len_cos - wid_sin, mid[1] + wid_cos - len_sin)
    bot_left  = (mid[0] + len_cos + wid_sin, mid[1] - wid_cos + len_sin)
    bot_right = (mid[0] - len_cos + wid_sin, mid[1] - wid_cos - len_sin)
    pts = (top_left, top_right, bot_right, bot_left)

    # draw and fill
    pygame.gfxdraw.aapolygon(window, pts, color)
    pygame.gfxdraw.filled_polygon(window, pts, color)

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
    width_px = consts.TRACK_WIDTH_M * consts.PX_METER_RATIO
    return tck, center_spline.T

def gen_inner_spline(tck, outer_spline):
    u_final = np.linspace(0, 1, num=200)
    width_px = consts.TRACK_WIDTH_M * consts.PX_METER_RATIO
    return __gen_offset_line(tck, u_final, outer_spline, -width_px)

#
# check if a line segment intersects with another segment
#
@njit
def __check_orientation(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

@njit
def do_segments_intersect(s1: np.ndarray, s2: np.ndarray) -> bool:
    # ref #1: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    # ref #2: https://stackoverflow.com/a/9997374
    # check orientation of lines against each other
    A, B = s1[0], s1[1]
    C, D = s2[0], s2[1]

    return (__check_orientation(A, C, D) != __check_orientation(B, C, D) and
            __check_orientation(A, B, C) != __check_orientation(A, B, D))

# return the intersection point of two segments
# ref: https://stackoverflow.com/a/565282
# all those multi lectures and linear algebra lessons to look this up... smh
@njit
def get_segment_intersection(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    A, B = s1[0], s1[1]
    C, D = s2[0], s2[1]

    # r & s are the vectors aligned with the direction of each segment
    # (A, B) is (A, A+r) -> B=A+r
    r, s = B-A, D-C

    # segments intersect at A + tr = B + us
    r_cross_x = cross2d(r, s)
    if r_cross_x == 0: return None # no intersection, prevent div by 0
    t = cross2d(C-A, s) / r_cross_x

    return (A + t*r)

@njit
def cross2d(A: np.ndarray, B: np.ndarray) -> float:
    return A[0] * B[1] - A[1] * B[0]

# #############################################
#
#               model tools
#
# #############################################

def create_model():
    # create model
    in_shape = (consts.NET_INPUT_SHAPE,)
    intlzr = lambda: initializers.RandomNormal(stddev=0.01)
    model = models.Sequential()
    model.add(layers.Dense(24, input_shape=in_shape, kernel_initializer=intlzr(), activation="relu"))
    model.add(layers.Dense(consts.NET_OUTPUT_SHAPE, kernel_initializer=intlzr(), activation="sigmoid"))

    # compile model
    model.compile(loss=losses.MeanSquaredError(),
                  metrics=consts.MODEL_METRICS,
                  optimizer=optimizers.Adam(learning_rate=consts.LEARNING_RATE))
    
    return model

# #############################################
#
#               misc. tools
#
# #############################################

@njit
def clamp(lower: float, upper: float, value: float) -> float:
    return max(min(value, upper), lower)

def avg(data: any) -> float:
    if type(data) is types.GeneratorType: # explode generator to list (~2x faster from testing than np.fromiter)
        data = list(data)

    if type(data) is np.ndarray: # faster for numpy arrays
        return np.sum(data) / data.shape[0]
    else: # tuples, lists, etc
        return sum(data) / len(data)

def get_reward_avg(rewards: any) -> float:
    weighted_total, weights_sum = 0, 0

    for reward in rewards:
        weighted_total += reward.value * reward.weight
        weights_sum += reward.weight

    if weights_sum == 0:
        return 0
    else:
        return weighted_total / weights_sum

@njit
def sign(n: float) -> float:
    return 1 if n > 0 else -1 if n < 0 else 0

def gen_model_folder():
    if not os.path.exists(consts.MODELS_FOLDER):
        os.mkdir(consts.MODELS_FOLDER)
    
    # get folder id (ref: https://stackoverflow.com/a/36150375)
    folder_id = hex( len(next(os.walk(consts.MODELS_FOLDER))[1]) + 1 )[2:]
    os.mkdir(consts.MODELS_FOLDER + folder_id)
    return consts.MODELS_FOLDER + folder_id