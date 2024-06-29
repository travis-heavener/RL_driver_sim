#
# window config
#
WIDTH  = 1600
HEIGHT =  900
FPS = 60
FPS_DISPLAY_RATE = 0.2 # in seconds, how often to refresh the frame rate display
PX_METER_RATIO = 1 # DEFAULT, updated by scaling; pixels per meter
def set_px_ratio(ratio: float) -> None:
    global PX_METER_RATIO
    PX_METER_RATIO = ratio

GRASS_COLOR_RGB     = ( 18, 118,  46)
TRACK_COLOR_RGB     = ( 96,  96,  96)
BARRIER_COLOR_RGB   = ( 15,  15,  15)
FINISH_COLOR_RGB    = (241, 241, 241)
DRIVELINE_COLOR_RGB = (255, 255,   0)
TEXT_COLOR_RGB      = (  5,   5,   5)
DEBUG_COLOR_RGB     = (200, 200, 255)
FONT_SIZE = HEIGHT // 40

USE_ANTIALIAS = True

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
LOWER_SHIFT_POINTS = (2000, 2700, 3400, 4100, 4650, 5050) # lowest RPMs for 2-7th gears

# sensors config
SHIFT_CONF_THRESH = 0.9 # how confident the model must be to shift up or down
SENSOR_RANGE_M = 200 # how far the sensors reach around the vehicle, in meters
SENSOR_ANGLES = (-75, -60, -45, -30, -20, -10, -5, -3, -1, 0, 1, 3, 5, 10, 20, 30, 45, 75, 60)
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
NET_INPUT_SHAPE = len(SENSOR_ANGLES) + 4 # n directions + speed, gear, rpms, horsepower (a la the butt dyno)
NET_OUTPUT_SHAPE = 6 # values from 0-1: accel, brake, left, right, downshift, upshift
LEARNING_RATE = 0.0005
BATCH_SIZE = 20
VALIDATION_SPLIT = 0.0 # [0, 1], how much training data should be used for validation data
TRAINING_EPOCHS = 1
MODEL_METRICS = ["accuracy"]
INTERPOLATION_FACTOR = 0.9 # how much of the rewards are factored into training data; from [0, 1]
TRAINING_EPSILON = 0.08 # if random is less than epsilon, a random move is done

MAX_GENERATION_TIME = 30 # in seconds, max lifetime of a generation before training
NUM_GENERATIONS = 200 # maximum number of generations