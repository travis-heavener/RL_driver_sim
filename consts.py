from math import sin, pi

#
# window config
#
WIDTH  = 1600
HEIGHT =  900
FPS = 60
PX_YARD_RATIO = 1 # pixels per yard
def set_px_yard_ratio(ratio: float) -> None:
    global PX_YARD_RATIO
    PX_YARD_RATIO = ratio

GRASS_COLOR_RGB   = ( 18, 118,  46)
TRACK_COLOR_RGB   = ( 96,  96,  96)
BARRIER_COLOR_RGB = ( 15,  15,  15)
FINISH_COLOR_RGB  = (241, 241, 241)

#
# model config
#
NET_INPUT_SHAPE = 17 # 12 directions, linear speed/accel, net force, engine rpms, current gear
NET_OUTPUT_SHAPE = 3 # steering [-1, 1] throttle [-1, 1], up/downshift [-1, 1]
LEARNING_RATE = 0.001

#
# vehicle config
#
DRIVER_IMG_PATH = "res/car.png"
VEHICLE_WEIGHT = 3200 # in lbs
VEHICLE_WIDTH_Y = 2 # in yards
VEHICLE_LENGTH_Y = 4 # in yards
ROLLING_DIAMETER = 25 # in inches

ENGINE_BRAKE_COEF = 0.06 # simulated crankshaft inertia for throttle
DRAG_COEF = 0.33 # estimated drag coefficient
ROLLING_FRICTION = 0.02 # coefficient of rolling friction (NOT kinetic friction)
GRAVITY_ACCEL = 9.81 # g, in m/s/s

# vehicle powertrain
# 375 lb-ft (@ 6019 RPMs)
# 468 hp (@ 7575 RPMs)
# top speed: 166.04 mph (7th gear @ 7812.83 RPMs, WOT)
MAX_TORQUE = 375 # in lb-ft
IDLE_RPMS, REDLINE_RPMS, MAX_RPMS = 750, 7000, 8000
FINAL_DRIVE = 4.43
GEAR_RATIOS = (3.50, 2.73, 2.13, 1.66, 1.30, 1.01, 0.79)

#
# track config
#
TRACK_WIDTH_Y = 8 # in yards
BARRIER_WIDTH_Y = 0.8 # in yards
START_WIDTH_Y = 3 # in yards

# the bounding box for where the track is to be placed
TRACK_BOUNDS = (
    (WIDTH * 0.00, HEIGHT * 0.10), # top-left
    (WIDTH * 0.75, HEIGHT * 0.95)  # bottom-right
)