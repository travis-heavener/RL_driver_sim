from math import sin, pi

#
# window config
#
WIDTH  = 1600
HEIGHT =  900
FPS = 60
PX_METER_RATIO = 1 # DEFAULT, updated by scaling; pixels per meter
def set_px_ratio(ratio: float) -> None:
    global PX_METER_RATIO
    PX_METER_RATIO = ratio

GRASS_COLOR_RGB   = ( 18, 118,  46)
TRACK_COLOR_RGB   = ( 96,  96,  96)
BARRIER_COLOR_RGB = ( 15,  15,  15)
FINISH_COLOR_RGB  = (241, 241, 241)

#
# vehicle config
#
DRIVER_IMG_PATH = "res/car.png"
VEHICLE_WEIGHT = 2800 # in lbs
VEHICLE_WIDTH_M = 2 # in meters
VEHICLE_LENGTH_M = 4 # in meters
ROLLING_DIAMETER_M = 0.635 # in meters

ENGINE_BRAKE_COEF = 0.06 # simulated crankshaft inertia for throttle
DRAG_COEF = 0.33 # estimated drag coefficient
ROLLING_FRICTION = 0.02 # coefficient of rolling friction (NOT kinetic friction)
BRAKING_FRICTION = 0.70 # coefficient of braking friction
GRAVITY_ACCEL = 9.81 # g, in m/s/s

# vehicle powertrain
STEERING_ANGLE = 60 # in degrees, maximum steering angle
MAX_SPEED = 75 # in m/s, governor-limited speed (set to a value near the vehicle's theoretical top speed)
MAX_TORQUE = 375 # in lb-ft
IDLE_RPMS, REDLINE_RPMS, MAX_RPMS = 750, 7000, 8000
FINAL_DRIVE = 4.43
GEAR_RATIOS = (3.50, 2.73, 2.13, 1.66, 1.30, 1.01, 0.79)

# sensors config
SHIFT_CONF_THRESH = 0.9 # how confident the model must be to shift up or down
SENSOR_RANGE_M = 200 # how far the sensors reach around the vehicle, in meters
SENSOR_ANGLES = (0, 30, 45, 60, 90, -30, -45, -60, -90)
SENSOR_NOT_FOUND = 1e6 # absurdly large number to indicate the sensor couldn't find anything nearby

#
# track config
#
TRACK_WIDTH_M = 8 # in meters
BARRIER_WIDTH_M = 0.8 # in meters
START_WIDTH_M = 3 # in meters

# the bounding box for where the track is to be placed
TRACK_BOUNDS = (
    (WIDTH * 0.05, HEIGHT * 0.05), # top-left
    (WIDTH * 0.95, HEIGHT * 0.95)  # bottom-right
)

#
# model config
#
NET_INPUT_SHAPE = len(SENSOR_ANGLES) + 3 # n directions + speed, gear, rpms
NET_OUTPUT_SHAPE = 3 # steering [-1, 1] throttle [-1, 1], up/downshift [-1, 1]
LEARNING_RATE = 0.01
BATCH_SIZE = 20
TRAINING_EPOCHS = 1
MODEL_METRICS = ["accuracy"]
INTERPOLATION_FACTOR = 0.8 # scalar applied to rewards
TRAINING_EPSILON = 0.125 # if random is less than epsilon, a random move is done

MAX_GENERATION_TIME = 30 # in seconds, max lifetime of a generation before training
NUM_GENERATIONS = 200 # maximum number of generations