# hide Tensorflow welcome prompt (thanks https://stackoverflow.com/a/38645250)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from collections import OrderedDict
import math
from numba import jit, njit
import numpy as np
from tensorflow import keras
from keras import layers, models, losses, optimizers, initializers
import pygame
from random import random
from time import time

import consts
import tools

# preload driver image & scale
DRIVER_IMG = pygame.image.load(consts.DRIVER_IMG_PATH)

#
# Used to describe driver behaviors which are factored into model decision evaluation
#
class DriverMood:
    # values: 0 to 1 (defaults below)
    accel       = 0.80  # higher values = faster acceleration
    braking     = 0.60  # higher values = later, harder braking
    speed       = 0.80  # higher values = more reward for higher speeds
    margin      = 0.50  # higher values = more distance between drivers and walls
    rev_bias    = 0.65  # higher values = bias to higher rpms (avoid 0 and 1 lol)

    def __init__(self, accel: float=None, braking: float=None, speed: float=None,
                       margin: float=None, rev_bias: float=None):
        self.accel = accel or self.accel
        self.braking = braking or self.braking
        self.speed = speed or self.speed
        self.margin = margin or self.margin
        self.rev_bias = rev_bias or self.rev_bias

# preset DriverMoods
MOOD_NORMAL   = DriverMood() # uses defaults
MOOD_SPEEDY   = DriverMood(accel=0.95, braking=0.80, speed=0.99, margin=0.30, rev_bias=0.90)
MOOD_YOLO     = DriverMood(accel=1.00, braking=0.90, speed=1.00, margin=0.10, rev_bias=0.99)
MOOD_NERVOUS  = DriverMood(accel=0.50, braking=0.40, speed=0.70, margin=0.85, rev_bias=0.35)
MOOD_ECO      = DriverMood(accel=0.35, braking=0.70, speed=0.65, margin=0.50, rev_bias=0.30)

#
# Used to aid in the calculation of rewards from driver states
#
class Reward:
    def __init__(self, value: float, weight: float):
        self.value = value # from -1 to 1
        self.weight = weight # >= 0

# preset reward weights
REWARD_LOW = 0.33
REWARD_MEDIUM = 0.67
REWARD_HIGH = 1

# #############################################
#
# Driver constructor for new, blank-slate driver models
#
# #############################################
_NEXT_DRIVER_ID = 1
THROTTLE, STEERING, SHIFTING = 0, 1, 2
class Driver:
    model: models.Sequential = None
    mood: DriverMood = MOOD_NORMAL
    img: pygame.Surface = None
    car_num: int
    has_crashed: bool = False
    memory: list = [] # store all states in this current generation's batch

    # physics
    x, y, width, length = 0.0, 0.0, 0, 0 # in meters
    speed, accel = 0.0, 0.0 # linear, scalar in m/s and m/s/s

    # engine & powertrain
    throttle: float = 0
    steering: float = 0
    direction: float = 0 # degrees
    gear: int = 1
    rpms: float = consts.IDLE_RPMS
    parking_start: float = -1 # how long the vehicle has gone without moving

    # reward system
    saved_state = None # the current state of the driver before the last move

    def __init__(self, mood: DriverMood=None):
        # update racer number
        global _NEXT_DRIVER_ID
        self.car_num = _NEXT_DRIVER_ID
        _NEXT_DRIVER_ID += 1

        # assign driver's mood
        self.mood = mood or self.mood

        # copy image
        self.width = math.ceil(consts.VEHICLE_WIDTH_M * consts.PX_METER_RATIO)
        self.length = math.ceil(consts.VEHICLE_LENGTH_M * consts.PX_METER_RATIO)
        self.img = pygame.transform.scale( DRIVER_IMG, (self.length, self.width) )

        # create new blank-slate network
        in_shape = (consts.NET_INPUT_SHAPE,)
        intlzr = lambda: initializers.RandomNormal(stddev=0.01)
        self.model = models.Sequential()
        self.model.add(layers.Dense(24, input_shape=in_shape, kernel_initializer=intlzr(), activation="linear"))
        self.model.add(layers.Dense(consts.NET_OUTPUT_SHAPE, kernel_initializer=intlzr(), activation=tools.scaled_tanh))

        self.model.compile(loss=losses.MeanSquaredError(),
                           metrics=consts.MODEL_METRICS,
                           optimizer=optimizers.Adam(learning_rate=consts.LEARNING_RATE))
    
    def reset(self):
        self.x = self._start_x
        self.y = self._start_y
        self.speed = 0
        self.accel = 0
        self.direction = 0
        
        self.throttle = 0
        self.steering = 0

        self.gear = 1; self.rpms = consts.IDLE_RPMS
        self.has_crashed = False
        self.parking_start = -1
        self.saved_state = None
        self.memory.clear()

    def set_start_pos(self, x: float, y: float) -> None:
        self.x = x; self.y = y
        self._start_x = x; self._start_y = y

    #
    # draws the driver on the window
    #
    def draw(self, window: pygame.Surface, draw_bbox: bool=False) -> None:
        if self.has_crashed: return

        img = pygame.transform.rotate(self.img, self.direction)
        center_pos = (self.x - self.length / 2, self.y - self.width / 2) # shifted back half-way to draw on middle
        center_rect = self.img.get_rect(topleft=center_pos).center
        pos = img.get_rect(center=center_rect)

        window.blit(img, pos)

        if draw_bbox:
            pygame.draw.lines(window, (0,0,255), False, self.bbox(closed=True), 2)

    #
    # enables the driver to make a decision, evaluate it, and update the model if they haven't crashed
    #
    def move(self, track_poly: np.ndarray, drivers: list[any], dt: float) -> None:
        if self.has_crashed: return

        # record state (if not saved from last evaluation--doesn't change)
        if self.saved_state is None:
            self.record_state(track_poly, drivers)
        
        # prepare data for model
        input_data = self._format_input_data(self.saved_state["state"])

        # run model
        output_data = None
        if random() <= consts.TRAINING_EPSILON:
            output_data = [(2 * random() - 1) for i in range(consts.NET_OUTPUT_SHAPE)]
        else:
            output_data = self.model(input_data, training=False)[0]

        # update telemetry
        throttle = round( float(output_data[THROTTLE]), 2 )
        steering = round( float(output_data[STEERING]), 2 )
        shift_confidence = round( float(output_data[SHIFTING]), 5 )

        if math.isnan(throttle) or math.isnan(steering) or math.isnan(shift_confidence):
            tools.warn("Model returned a nan value, aborting move...:", output_data)
            return
        
        self.throttle, self.steering = throttle, steering

        # update physical properties
        net_force = tools.calcVehicleForce(self.gear, self.rpms, self.throttle)
        self.accel = net_force / tools.lbs2kg(consts.VEHICLE_WEIGHT)

        # check for parking status
        if self.speed == 0:
            self.rpms = consts.IDLE_RPMS

            if self.throttle >= 0: # if stopped and not braking, start moving
                self.speed = tools.getSpeedFromRPMs(self.gear, self.rpms)
            elif self.parking_start == -1:
                self.parking_start = time()


        # electronically limit speed
        self.speed += self.accel * dt
        self.speed = max(0, min(self.speed, consts.MAX_SPEED)) # govern speed

        # allow steering if not parked
        if not self.is_parked():
            # -1 is left, +1 is right
            self.direction -= self.steering * consts.STEERING_ANGLE * dt

        speed_scaled = self.speed * dt * consts.PX_METER_RATIO
        angle_rad = math.radians(self.direction)
        self.x += math.cos(angle_rad) * speed_scaled
        self.y -= math.sin(angle_rad) * speed_scaled

        # update rpms if accelerating from stop or already moving
        if not self.is_parked():
            self.parking_start = -1
            self.rpms = max(consts.IDLE_RPMS, tools.getRPMsFromSpeed(self.speed, self.gear))
        else:
            self.accel = 0.0

        # shift gears last to make calculations easier
        if shift_confidence > consts.SHIFT_CONF_THRESH and self.gear < len(consts.GEAR_RATIOS): # upshift
            self.gear += 1
        elif shift_confidence < -consts.SHIFT_CONF_THRESH and self.gear > 1: # downshift
            self.gear -= 1

        # record state's output
        self.saved_state["output"] = output_data

        # force telemetry data to be Python floats
        if type(self.x) is not float: self.x = float(self.x)
        if type(self.y) is not float:  self.y = float(self.y)
        if type(self.speed) is not float: self.speed = float(self.speed)
        if type(self.accel) is not float: self.accel = float(self.accel)
        if type(self.direction) is not float: self.direction = float(self.direction)
        if type(self.throttle) is not float: self.throttle = float(self.throttle)
        if type(self.steering) is not float: self.steering = float(self.steering)

        # print(f"Speed: {round(tools.ms2mph(self.speed))} mph, Gear: {self.gear}, RPMs: {round(self.rpms)}, Throttle: {self.throttle}")

    #
    # get the driver's current state as a dict
    #
    def record_state(self, track_poly: np.ndarray, drivers: list[any]) -> OrderedDict:
        if self.saved_state is None:
            self.saved_state = {"state": OrderedDict(), "output": None}
        
        # get sensor data
        sensor_data = self._scan_sensors(track_poly, drivers)
        for data in sensor_data:
            self.saved_state["state"][data[0]] = data[1] # angle, distance pairs

        # record other telemetry
        self.saved_state["state"]["speed"] = self.speed
        self.saved_state["state"]["gear"] = self.gear
        self.saved_state["state"]["rpms"] = self.rpms
    
    #
    # used to transform the current state to training data for the model
    #
    def _format_input_data(self, state: OrderedDict) -> np.ndarray:
        return np.array( list(state.values()) ).reshape(1, -1).astype(np.float32)

    #
    # evaluate the driver's last move
    #
    def evaluate(self, track_poly: np.ndarray, drivers: list[any]) -> None:
        # get current state
        reward = self._get_reward(track_poly, drivers)
        input_data = self._format_input_data(self.saved_state["state"])
        self.memory.append({"input": input_data, "output": self.saved_state["output"], "reward": reward})

        # reset state
        self.saved_state = None

    #
    # returns True if parked
    #
    def is_parked(self) -> bool:
        return self.speed == 0 and self.throttle < 0

    #
    # searches for any obstacles at the surrounding angles
    # returns a list of (angle, distance) pairs for each direction
    #
    def _scan_sensors(self, track_poly: np.ndarray, drivers: list[any]) -> tuple[tuple[float, float]]:
        bboxes = np.array([
            driver.bbox() for driver in drivers if driver.car_num != self.car_num and not driver.has_crashed
        ])

        # extract all segments for intersection checking
        obstacle_segs = tools.get_obstacle_segs(track_poly, bboxes)

        sensor_data = []
        max_range = consts.SENSOR_RANGE_M * consts.PX_METER_RATIO
        pos = np.array([self.x, self.y])
        for angle in consts.SENSOR_ANGLES:
            # calculate end point
            angle_rad = math.radians(self.direction + angle)
            u = np.array([math.cos(angle_rad), math.sin(angle_rad)]) # unit vector
            ray = np.array([pos, pos + u * max_range])

            # start concurrent process
            sensor_data.append( (angle, tools.emit_ray(obstacle_segs, ray)) )

        return sensor_data

    #
    # Get the bounding box around the vehicle of its vertices
    #
    def bbox(self, closed=False):
        # NOTE #1: the driver is drawn so that self.x and self.y are the CENTER of the image
        # NOTE #2: unrotated vehicle is wider than taller, so its height *is* self.width
        x, y, width, height = self.x, self.y, self.length, self.width
        corners = np.array((
            (x - width/2, y - height/2), (x + width/2, y - height/2),
            (x + width/2, y + height/2), (x - width/2, y + height/2)
        ))

        # rotate bbox
        theta = math.radians(-self.direction)
        rot_matrix = np.array((
            (math.cos(theta), -math.sin(theta)),
            (math.sin(theta), math.cos(theta))
        ))

        rotated_bbox = []
        pos = np.array((self.x, self.y))
        for corner in corners:
            rotated_bbox.append( np.dot(rot_matrix, corner - pos) + pos )

        if closed: rotated_bbox.append( rotated_bbox[0] )

        return np.array(rotated_bbox)

    #
    # Returns a tuple of the lower and upper shift points for the current gear,
    # factoring in driver mood.
    #
    def get_shift_points(self) -> tuple[int, int]:
        lower = None if self.gear == 1 else consts.LOWER_SHIFT_POINTS[self.gear-2]
        if lower is not None:
            lower -= 0.67 * self.mood.rev_bias * (consts.MAX_RPMS - consts.REDLINE_RPMS)
            lower = max(lower, consts.IDLE_RPMS)
        upper = consts.MAX_RPMS - 2 * self.mood.rev_bias * (consts.MAX_RPMS - consts.REDLINE_RPMS)
        return (lower, upper)

    #
    # Calculate the driver's reward from their current position based on their mood
    # Note:
    #     While the network's outputs may be ranged from [-1, 1] indicating various things,
    #     these rewards do NOT indicate the same ideas. The values in this function indicate
    #     a rating of bad (-1) to good (+1) for the given output.
    #     
    #     Ex. -0.75 steering means to turn left from the model;
    #         here, it means that the move was 75% unfavorable.
    #
    def _get_reward(self, track_poly: np.ndarray, drivers: list[any]) -> list[float]:
        rewards: list[list[Reward]] = [ [], [], [] ]
        SENSOR_RANGE = consts.SENSOR_RANGE_M

        ####################################
        ########## CRASH CHECKING ##########
        ####################################

        # check for collisions
        self._collision_check(track_poly, drivers)

        if self.has_crashed:
            rewards[THROTTLE].append( Reward(-0.5, REWARD_HIGH) )
            rewards[STEERING].append( Reward(-1, REWARD_HIGH) )
            return [ tools.get_reward_avg(reward) for reward in rewards ]

        # check for money shifts
        if self.rpms >= consts.MAX_RPMS:
            self.has_crashed = True
            rewards[THROTTLE].append( Reward(-0.5, REWARD_HIGH) )
            rewards[SHIFTING].append( Reward(-1, REWARD_HIGH) )
            return [ tools.get_reward_avg(reward) for reward in rewards ]

        """ UNUSED
        # check for sharp steering
        # literally just roll die to see whether or not the vehicle crashed
        flip_thresh = random()
        probability = 0.5 * self.steering / consts.STEERING_ANGLE * (np.tanh(self.speed / 5 - 4) + 1)
        self.has_crashed = flip_thresh < probability

        if self.has_crashed:
            rewards[THROTTLE] = -3.5; rewards[STEERING] = -5; rewards[SHIFTING] = -2.5
            return rewards
        """

        ####################################
        ########## STATE CHECKING ##########
        ####################################

        ######### get current state & relevant metrics #########
        state_i = self.saved_state["state"]
        self.record_state(track_poly, drivers)
        state_f = self.saved_state["state"]

        # avg distance from all sensors, either front, left, or right groupings
        # from [0, 1], 0 being no gap and 1 being massive gap
        front_gap_i = tools.avg(min(state_i[a] / SENSOR_RANGE, 1) for a in consts.SENSOR_ANGLES if abs(a) <= 10)
        left_gap_i  = tools.avg(min(state_i[a] / SENSOR_RANGE, 1) for a in consts.SENSOR_ANGLES if a < -10)
        right_gap_i = tools.avg(min(state_i[a] / SENSOR_RANGE, 1) for a in consts.SENSOR_ANGLES if a > 10)
        
        front_gap_f = tools.avg(min(state_f[a] / SENSOR_RANGE, 1) for a in consts.SENSOR_ANGLES if abs(a) <= 10)
        left_gap_f  = tools.avg(min(state_f[a] / SENSOR_RANGE, 1) for a in consts.SENSOR_ANGLES if a < -10)
        right_gap_f = tools.avg(min(state_f[a] / SENSOR_RANGE, 1) for a in consts.SENSOR_ANGLES if a > 10)
        
        # from [-1, 1], the change in gap after the last move
        delta_front_gap = front_gap_f - front_gap_i
        delta_left_gap  = left_gap_f - left_gap_i
        delta_right_gap = right_gap_f - right_gap_i

        # acceleration & shifting changes
        # has_accelerated is NOT the same as is_accelerating
        #   ex. the throttle can be open while slowing down
        has_accelerated = state_f["speed"] > state_i["speed"]
        is_accelerating = self.throttle > 0
        is_braking = self.throttle < 0
        has_upshifted = state_f["gear"] > state_i["gear"]
        has_downshifted = state_f["gear"] < state_i["gear"]

        ######### reward evaluation for state changes #########
        
        # evaluate accelerating and slowing near gaps
        does_gap_exist = front_gap_f >= self.mood.margin # based on driver mood
        gap_reward = -1 if not has_accelerated else 1
        
        if does_gap_exist:
            # reward for accelerating, punishment for slowing
            gap_reward *= 1 + ((front_gap_f - 1) / max(1 - self.mood.margin, 0.01)) ** 3
        else:
            # reward for slowing, punishment for accelerating
            gap_reward *= -1 + (front_gap_f / max(self.mood.margin, 0.01)) ** (1/3)
        
        rewards[THROTTLE].append( Reward(gap_reward, REWARD_HIGH) )

        # handle if steering towards or away from side obstacles (steering)
        # allow for 0.25 - 1.25 vehicle widths of space on either side, factoring in mood margin
        front_gap_thresh = 3 * self.mood.margin * consts.VEHICLE_LENGTH_M / SENSOR_RANGE
        side_gap_thresh = (self.mood.margin + 0.25) * consts.VEHICLE_WIDTH_M / SENSOR_RANGE
        
        def reward_side_gap(gap, delta):
            if gap < side_gap_thresh and delta < 0: # indicate bad maneuver
                scaled_gap = gap / side_gap_thresh
                rewards[STEERING].append( Reward(-0.25 - 0.75 * scaled_gap, REWARD_HIGH) )

                # indicate poor accelerating or good braking
                rewards[THROTTLE].append( Reward((1 - scaled_gap) * -self.throttle, REWARD_HIGH) )
            elif delta > 0: # reward for opening up space
                rewards[STEERING].append( Reward(delta, REWARD_HIGH) )

        # handle lateral spacing
        reward_side_gap(left_gap_f, delta_left_gap)
        reward_side_gap(right_gap_f, delta_right_gap)

        # handle if heading towards obstacles (throttle)
        if front_gap_f < front_gap_thresh and delta_front_gap < 0: # not enough space in front
            scaled_gap = front_gap_f / front_gap_thresh
            rewards[THROTTLE].append( Reward((1 - scaled_gap) * -self.throttle, REWARD_HIGH) )
        
        # reward and punish shift points
        shift_points = self.get_shift_points()
        scaled_rpms = self.rpms / consts.MAX_RPMS # rpms from 0 to 1
        lower_pt = shift_points[0] / consts.MAX_RPMS if shift_points[0] is not None else None
        upper_pt = shift_points[1] / consts.MAX_RPMS
        shift_reward = 0

        if has_upshifted:
            if scaled_rpms < lower_pt:
                shift_reward = -(lower_pt - scaled_rpms) / lower_pt
            else:
                shift_reward = 1 - (scaled_rpms - lower_pt) / consts.MAX_RPMS
        elif has_downshifted:
            if scaled_rpms > upper_pt:
                upper_rpms_thresh = consts.MAX_RPMS - upper_pt
                shift_reward = -(scaled_rpms - upper_pt) / upper_rpms_thresh
            else:
                shift_reward = scaled_rpms / upper_pt

        rewards[SHIFTING].append( Reward(shift_reward, REWARD_HIGH) )

        ####################################
        ######### STATELESS CHECKS #########
        ####################################

        # determine if the driver is oriented with the driveline
        # TODO

        # determine if the throttle is open (*trying* to accel)
        if is_accelerating: # reward for higher rpms; punish for lower rpms
            accel_rpms_reward = 1 - 2 * abs(scaled_rpms - self.mood.accel)
            rewards[THROTTLE].append( Reward(accel_rpms_reward, REWARD_MEDIUM) )
        elif is_braking: # reward for higher rpms
            brake_rpms_reward = 1 - abs(scaled_rpms - self.mood.braking)
            rewards[THROTTLE].append( Reward(brake_rpms_reward, REWARD_MEDIUM) )

        # reward drivers for moving faster
        if is_accelerating:
            speed_reward = 2 * (self.speed / consts.MAX_SPEED) - 1
            rewards[THROTTLE].append( Reward(speed_reward, REWARD_MEDIUM) )

        # punish drivers for being parked
        if self.parking_start != -1:
            time_parked = time() - self.parking_start
            rewards[THROTTLE].append( Reward(-time_parked, REWARD_HIGH) )

        # average out rewards
        return [ tools.get_reward_avg(reward) for reward in rewards ]

    #
    # check for collisions with the track or players
    #
    def _collision_check(self, track_poly: np.ndarray, drivers: list[any]):
        bboxes = np.array([
            driver.bbox() for driver in drivers if driver.car_num != self.car_num and not driver.has_crashed
        ])
        self_bbox = self.bbox()

        # extract all segments for intersection checking
        obstacle_segs = tools.get_obstacle_segs(track_poly, bboxes)

        # for all edges of this bbox, treat as rays to passthru raycast method as shortcut
        # since the raycast method checks for intersections between line segments
        self_bbox_len = len(self_bbox)
        for i in range(self_bbox_len):
            ray = np.array([self_bbox[i], self_bbox[(i+1) % self_bbox_len]])
            dist = tools.emit_ray(obstacle_segs, ray, return_on_hit=True)
            if dist != consts.SENSOR_NOT_FOUND:
                self.has_crashed = True
                return
    
    #
    # train the model and clear the memory bank
    #
    def train(self):
        # if a driver is crashed into from the start, ignore them from training
        if len(self.memory) == 0:
            tools.warn(f"Driver {self.car_num} crashed before logging memories, cannot train.")
            return

        # extract data from bank
        inputs, outputs, rewards = [], [], []
        for memory in self.memory:
            inputs.append(memory["input"])
            outputs.append(memory["output"])
            rewards.append(memory["reward"])

        # calculate target data
        target_outputs = []
        
        for output, reward in zip(outputs, rewards):
            target_outputs.append([])
            for i in range(consts.NET_OUTPUT_SHAPE):
                offset = tools.sign(float(output[i])) * reward[i] * consts.INTERPOLATION_FACTOR
                target_outputs[-1].append( output[i] + offset )
    
        # compile training data
        training_x = np.array(inputs)[:,0,:].astype(np.float32)
        training_y = np.array(target_outputs).astype(np.float32)
        
        # train model
        self.model.fit(training_x, training_y,
                       batch_size=consts.BATCH_SIZE,
                       epochs=consts.TRAINING_EPOCHS,
                       verbose=1)

        # wipe memory bank
        self.memory.clear()

# #############################################
#
# TrainedDriver for loading existing models
#
# #############################################
class TrainedDriver(Driver):
    pass