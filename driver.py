# hide Tensorflow welcome prompt (thanks https://stackoverflow.com/a/38645250)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from collections import OrderedDict
import math
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, losses, optimizers, initializers
import pygame
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
        initializer = lambda: initializers.RandomNormal(stddev=0.01)
        self.model = models.Sequential()
        self.model.add(layers.Dense(24, input_shape=in_shape, kernel_initializer=initializer(), activation="linear"))
        self.model.add(layers.Dense(consts.NET_OUTPUT_SHAPE, kernel_initializer=initializer(), activation=tools.scaled_tanh))

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

        # prepare data for model
        state = self.get_state(track_poly, drivers)
        input_data = self._format_input_data(state)

        # record state
        self.saved_state = {"state": state, "output": None}

        # run model
        output_data = None
        if np.random.random() <= consts.TRAINING_EPSILON:
            output_data = [(2 * np.random.random() - 1) for i in range(consts.NET_OUTPUT_SHAPE)]
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
            self.direction += self.steering * consts.STEERING_ANGLE * dt

        speed_scaled = self.speed * dt * consts.PX_METER_RATIO
        angle_rad = np.deg2rad(self.direction)
        self.x += np.cos(angle_rad) * speed_scaled
        self.y -= np.sin(angle_rad) * speed_scaled

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

    #
    # get the driver's current state as a dict
    #
    def get_state(self, track_poly: np.ndarray, drivers: list[any]) -> OrderedDict:
        state = OrderedDict()
        
        # get sensor data
        sensor_data = self._scan_sensors(track_poly, drivers)
        for data in sensor_data:
            state[data[0]] = data[1] # angle, distance pairs

        # record other telemetry
        state["speed"] = self.speed
        state["gear"] = self.gear
        state["rpms"] = self.rpms
        return state
    
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
        bboxes = [driver.bbox() for driver in drivers if driver.car_num != self.car_num and not driver.has_crashed]

        sensor_data = []
        max_range = consts.SENSOR_RANGE_M * consts.PX_METER_RATIO
        pos = np.array((self.x, self.y))
        for angle in consts.SENSOR_ANGLES:
            # calculate end point
            angle_rad = np.deg2rad(self.direction + angle)
            u = np.array((np.cos(angle_rad), np.sin(angle_rad))) # unit vector
            ray = (pos, pos + u * max_range)

            sensor_data.append( (angle, self._emit_ray(track_poly, bboxes, ray)) )

        return sensor_data

    #
    # searches for the nearest obstacle from the given angle out
    # returns the distance to the nearest obstacle
    #
    def _emit_ray(self, track_poly: np.ndarray, bboxes: list[np.ndarray], ray: tuple[float, float]) -> float:
        closest_obstacle = consts.SENSOR_NOT_FOUND

        # 1. check for intersections with the track
        len_track_poly = len(track_poly)
        for i in range(len_track_poly):
            # skip implicit segments (that connect outer to inner walls & vice versa)
            # middle 2 pts connect outer to inner, outer 2 pts connect inner to outer
            if i == len_track_poly-1 or i+1 == len_track_poly // 2: continue

            # cast ray
            seg = (track_poly[i], track_poly[(i+1) % len_track_poly])
            if tools.do_segments_intersect(ray, seg):
                intersection = tools.get_segment_intersection(ray, seg)
                if intersection is None: continue

                dist_m = np.hypot(*(intersection - ray[0])) / consts.PX_METER_RATIO
                if dist_m < closest_obstacle:
                    closest_obstacle = dist_m

        # 2. check for intersections with vehicles
        for bbox in bboxes:
            len_bbox = len(bbox)
            for i in range(len_bbox):
                # cast ray
                seg = (bbox[i], bbox[(i+1) % len_bbox])
                if tools.do_segments_intersect(ray, seg):
                    intersection = tools.get_segment_intersection(ray, seg)
                    if intersection is None: continue

                    dist_m = np.hypot(*(intersection - ray[0])) / consts.PX_METER_RATIO
                    if dist_m < closest_obstacle:
                        closest_obstacle = dist_m

        return closest_obstacle

    #
    # get the bounding box around the vehicle of its vertices
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
        theta = np.deg2rad(-self.direction)
        rot_matrix = np.array((
            (np.cos(theta), -np.sin(theta)),
            (np.sin(theta), np.cos(theta))
        ))

        rotated_bbox = []
        pos = np.array((self.x, self.y))
        for corner in corners:
            rotated_bbox.append( np.dot(rot_matrix, corner - pos) + pos )

        if closed: rotated_bbox.append( rotated_bbox[0] )

        return np.array(rotated_bbox)

    #
    # calculate the driver's reward from their current position based on their mood
    #
    def _get_reward(self, track_poly: np.ndarray, drivers: list[any]) -> list[float]:
        rewards = [0, 0, 0]

        ####################################
        ########## CRASH CHECKING ##########
        ####################################

        # check for collisions
        self._collision_check(track_poly, drivers)

        if self.has_crashed:
            rewards[THROTTLE] = -1; rewards[STEERING] = -5; rewards[SHIFTING] = -1
            return rewards

        # check for money shifts
        if self.rpms >= consts.MAX_RPMS:
            self.has_crashed = True
            rewards[THROTTLE] = -3; rewards[SHIFTING] = -5
            return rewards

        """ UNUSED
        # check for sharp steering
        # literally just roll die to see whether or not the vehicle crashed
        flip_thresh = np.random.random()
        probability = 0.5 * self.steering / consts.STEERING_ANGLE * (np.tanh(self.speed / 5 - 4) + 1)
        self.has_crashed = flip_thresh < probability

        if self.has_crashed:
            rewards[THROTTLE] = -3.5; rewards[STEERING] = -5; rewards[SHIFTING] = -2.5
            return rewards
        """
        ####################################
        ########## STATE CHECKING ##########
        ####################################

        # get current state
        current_state = self.get_state(track_poly, drivers)
        prev_state = self.saved_state["state"]

        # determine if obstacles ahead are closer
        are_obstacles_closer = False 

        # determine if the driver is now closer to obstacles or not
        delta_margin = 0 # the change in proximity of obstacles nearby
        for key in current_state.keys():
            if key in consts.SENSOR_ANGLES:
                last_dist = min(prev_state[key], consts.SENSOR_RANGE_M) # clamp off SENSOR_NOT_FOUND
                current_dist = min(current_state[key], consts.SENSOR_RANGE_M) # clamp off SENSOR_NOT_FOUND
                delta_margin += (current_dist - last_dist) / consts.SENSOR_RANGE_M # from [-1, 1]
        delta_margin /= len(consts.SENSOR_ANGLES) # avg across all sensors to clamp from [-1, 1]
        delta_margin = np.sign(delta_margin) # -1 is CLOSER to obstacles, 0 is same, 1 is FURTHER

        # reward drivers for their proximity to obstacles
        rewards[THROTTLE] += delta_margin * 0.75
        rewards[STEERING] += delta_margin

        # reward for throttle (acceleration or braking) closer to their accel/braking mood
        if self.throttle > 0:
            rewards[THROTTLE] += 0.5 - np.abs(self.throttle - self.mood.accel)
        else:
            rewards[THROTTLE] += 0.5 - np.abs(-self.throttle - self.mood.braking)

        # reward drivers for their overall speed
        speed_reward = 0.5 - np.abs((self.speed / consts.MAX_SPEED) - self.mood.speed)
        rewards[THROTTLE] += speed_reward
        rewards[STEERING] += speed_reward * 0.2
        rewards[SHIFTING] += speed_reward * 0.8

        # reward drivers for speeding up during a gap/opening, OR
        # punish drivers for slowing down during a gap/opening
        speed_diff = (current_state["speed"] - prev_state["speed"]) / consts.MAX_SPEED # [-1, 1]
        if speed_diff > 0:
            gap_reward = np.sign(speed_diff) * speed_diff
            rewards[THROTTLE] += gap_reward
            rewards[SHIFTING] += gap_reward

        # reward drivers for their available space around themselves
        # average all sensors' readings
        # sensor_data = self._scan_sensors(track_poly, drivers)
        # obstacle_reward = 0
        # for reading in sensor_data:
        #     speed_ratio = self.speed / consts.MAX_SPEED # prefer angles closer to ahead when faster
        #     scaled_reading = min(reading[1] / consts.SENSOR_RANGE_M, 1) # from [0, 1]
            
        #     obstacle_reward += 0.5 - np.abs(scaled_reading - speed_ratio)
        # obstacle_reward /= len(sensor_data) # from [-0.5, 0.5]

        # rewards[THROTTLE] += obstacle_reward * 0.25
        # rewards[STEERING] += obstacle_reward

        # reward drivers for their rpms' proximity to target [-0.5, 0.5]
        rpm_reward = 0.5 - np.abs((self.rpms / consts.MAX_RPMS) - self.mood.rev_bias)
        rewards[THROTTLE] += rpm_reward * 0.25
        rewards[SHIFTING] += rpm_reward
        

        # punish drivers for being parked
        if self.parking_start != -1:
            time_parked = time() - self.parking_start
            rewards[THROTTLE] -= time_parked
            rewards[SHIFTING] -= time_parked * 0.5

        return rewards

    #
    # check for collisions with the track or players
    #
    def _collision_check(self, track_poly: np.ndarray, drivers: list[any]):
        bboxes = [driver.bbox() for driver in drivers if driver.car_num != self.car_num and not driver.has_crashed]
        self_bbox = self.bbox()

        # for all edges of this bbox, treat as rays to passthru raycast method as shortcut
        # since the raycast method checks for intersections between line segments
        self_bbox_len = len(self_bbox)
        for i in range(self_bbox_len):
            ray = (self_bbox[i], self_bbox[(i+1) % self_bbox_len])
            dist = self._emit_ray(track_poly, bboxes, ray)
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
        
        num_outputs = len(outputs[0])
        for output, reward in zip(outputs, rewards):
            target_outputs.append([])
            for i in range(num_outputs):
                target_outputs[-1].append( output[i] + np.sign(output[i]) * reward[i] * consts.INTERPOLATION_FACTOR )
    
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