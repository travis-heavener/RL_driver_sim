# hide Tensorflow welcome prompt (thanks https://stackoverflow.com/a/38645250)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from collections import OrderedDict
import math
from numba import jit, njit
import numpy as np
import tensorflow as tf
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
MOOD_SPEEDY   = DriverMood(accel=1.00, braking=0.90, speed=0.99, margin=0.30, rev_bias=0.90)
MOOD_YOLO     = DriverMood(accel=1.00, braking=1.00, speed=1.00, margin=0.10, rev_bias=1.00)
MOOD_NERVOUS  = DriverMood(accel=0.50, braking=0.40, speed=0.70, margin=0.85, rev_bias=0.35)
MOOD_ECO      = DriverMood(accel=0.35, braking=0.70, speed=0.65, margin=0.50, rev_bias=0.30)

# #############################################
#
# Driver constructor for new, blank-slate driver models
#
# #############################################
_NEXT_DRIVER_ID = 1
ACCEL, BRAKE, LEFT, RIGHT, DOWNSHIFT, UPSHIFT = range(6)
class Driver:
    model: tf.keras.models.Sequential = None
    mood: DriverMood = MOOD_NORMAL
    img: pygame.Surface = None
    car_num: int
    has_crashed: bool = False
    pos_restore_status: bool = False # whether or not to restore the driver's telemetry between generations
    memory: list = [] # store all states in this current generation's batch

    # physics
    x, y, width, length = 0.0, 0.0, 0, 0 # in meters
    speed, accel = 0.0, 0.0 # linear, scalar in m/s and m/s/s

    # engine & powertrain
    throttle: float = 0 # accel - brake
    steering: float = 0 # right - left
    direction: float = 0 # degrees
    gear: int = 1
    rpms: float = consts.IDLE_RPMS

    # reward system
    saved_state = None # the current state of the driver before the last move

    def __init__(self, mood: DriverMood=None):
        # update racer number
        global _NEXT_DRIVER_ID
        self.car_num = _NEXT_DRIVER_ID
        _NEXT_DRIVER_ID += 1

        self.mood = mood or self.mood # assign driver's mood

        # copy image
        self.width = math.ceil(consts.VEHICLE_WIDTH_M * consts.PX_METER_RATIO)
        self.length = math.ceil(consts.VEHICLE_LENGTH_M * consts.PX_METER_RATIO)
        self.img = pygame.transform.scale( DRIVER_IMG, (self.length, self.width) )

        # create new blank-slate network
        self.model = tools.create_model()
    
    def log(self):
        print(f"Speed: {round(tools.ms2mph(self.speed))} mph, Gear: {self.gear}, RPMs: {round(self.rpms)}, Throttle: {self.throttle}")

    def reset(self):
        if not self.has_crashed and not self.pos_restore_status: return

        self.x, self.y = self._start_x, self._start_y
        self.speed = self.accel = self.direction = 0
    
        self.throttle = self.steering = 0
        self.gear, self.rpms = 1, consts.IDLE_RPMS

        self.saved_state["state"].clear()
        self.saved_state["output"] = None
        self.memory.clear()
        self.has_crashed = False

    def set_start_pos(self, x: float, y: float) -> None:
        self.x = x; self.y = y
        self._start_x = x; self._start_y = y

    def set_pos_restore(self, status: bool) -> None:
        self.pos_restore_status = status

    #
    # draws the driver on the window
    #
    def draw(self, window: pygame.Surface, draw_bbox: bool=False, draw_sensor_paths=False) -> None:
        if self.has_crashed: return

        img = pygame.transform.rotate(self.img, self.direction)
        center_pos = (self.x - self.length / 2, self.y - self.width / 2) # shifted back half-way to draw on middle
        center_rect = self.img.get_rect(topleft=center_pos).center
        pos = img.get_rect(center=center_rect)

        window.blit(img, pos)

        # draw bbox
        if draw_bbox:
            pygame.draw.lines(window, consts.DEBUG_COLOR_RGB, False, self.bbox(closed=True), 2)
        
        # draw sensor object recognition paths
        if draw_sensor_paths and self.saved_state is not None:
            pos = np.array([self.x, self.y])
            for angle in consts.SENSOR_ANGLES:
                if angle not in self.saved_state["state"].keys() or self.saved_state["state"][angle] == consts.SENSOR_NOT_FOUND:
                    continue

                # calculate end point
                angle_rad = math.radians(self.direction + angle)
                u = np.array([math.cos(angle_rad), -math.sin(angle_rad)]) # unit vector
                offset = u * math.cos(math.radians(angle)) * consts.VEHICLE_LENGTH_M * consts.PX_METER_RATIO / 2
                dist = self.saved_state["state"][angle] * consts.PX_METER_RATIO
                ray = np.array([pos + offset, pos + u * dist + offset])
                pygame.draw.line(window, consts.DEBUG_COLOR_RGB, *ray, 2)

    #
    # enables the driver to make a decision, evaluate it, and update the model if they haven't crashed
    #
    def move(self, track_poly: np.ndarray, drivers: list[any], dt: float) -> None:
        if self.has_crashed: return

        self.record_state(track_poly, drivers) # record state
        input_data = self._format_input_data(self.saved_state["state"]) # prepare data for model

        # run model
        output_data = None
        if random() <= consts.TRAINING_EPSILON:
            output_data = [random() for i in range(consts.NET_OUTPUT_SHAPE)]
        else:
            output_data = self.model(input_data, training=False)[0]

        # update telemetry
        normalize = lambda n: round( float(n), 2 )
        self.throttle = normalize(output_data[ACCEL] - output_data[BRAKE]) # [-1, 1]
        self.steering = normalize(output_data[RIGHT] - output_data[LEFT]) # [-1, 1]
        downshift_conf = normalize(output_data[DOWNSHIFT])
        upshift_conf = normalize(output_data[UPSHIFT])

        # update physical properties
        net_force = tools.calcVehicleForce(self.gear, self.rpms, self.throttle)
        self.accel = net_force / tools.lbs2kg(consts.VEHICLE_WEIGHT)

        # check for parking status
        if self.speed == 0:
            self.rpms = consts.IDLE_RPMS

            if self.throttle >= 0: # if stopped and not braking, start moving
                self.speed = tools.rpms_to_speed(self.gear, self.rpms)


        # electronically limit speed
        self.speed += self.accel * dt
        self.speed = max(0, min(self.speed, consts.MAX_SPEED)) # govern speed

        # handle parking status
        if not self.is_parked():
            # allow steering if not parked
            self.direction -= self.steering * consts.STEERING_ANGLE * dt # -1 is left, +1 is right

            # update rpms if accelerating from stop or already moving
            self.rpms = max(consts.IDLE_RPMS, tools.speed_to_rpms(self.speed, self.gear))
        else:
            self.accel = 0.0

        speed_scaled = self.speed * dt * consts.PX_METER_RATIO
        angle_rad = math.radians(self.direction)
        self.x += math.cos(angle_rad) * speed_scaled
        self.y -= math.sin(angle_rad) * speed_scaled

        # shift gears last to make calculations easier
        is_downshifting = downshift_conf > consts.SHIFT_CONF_THRESH
        is_upshifting = upshift_conf > consts.SHIFT_CONF_THRESH

        if is_upshifting and not is_downshifting and self.gear < len(consts.GEAR_RATIOS):
            self.gear += 1 # upshift
        elif is_downshifting and not is_upshifting and self.gear > 1:
            self.gear -= 1 # downshift

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
    def record_state(self, track_poly: np.ndarray, drivers: list[any]) -> OrderedDict:
        if self.saved_state is None:
            self.saved_state = {"state": OrderedDict(), "output": None}
        
        # reset state
        self.saved_state["state"].clear()
        self.saved_state["output"] = None
        
        # get sensor data
        sensor_data = self._scan_sensors(track_poly, drivers)
        for data in sensor_data:
            self.saved_state["state"][data[0]] = data[1] # angle, distance pairs

        # record other telemetry
        self.saved_state["state"]["speed"] = self.speed
        self.saved_state["state"]["gear"] = self.gear
        self.saved_state["state"]["rpms"] = self.rpms
        self.saved_state["state"]["horsepower"] = tools.get_engine_HP(self.rpms, self.throttle)
    
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
            u = np.array([math.cos(angle_rad), -math.sin(angle_rad)]) # unit vector
            offset = u * math.cos(math.radians(angle)) * consts.VEHICLE_LENGTH_M * consts.PX_METER_RATIO / 2
            ray = np.array([pos + offset, pos + u * max_range + offset])

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
    # Calculate the driver's reward from their current position based on their mood.
    # The rewards' values correspond to what the model's output values should have been [0, 1].
    #
    def _get_reward(self, track_poly: np.ndarray, drivers: list[any]) -> list[float]:
        rewards = [ 0.5 for i in range(consts.NET_OUTPUT_SHAPE) ]
        rewards[UPSHIFT] = rewards[DOWNSHIFT] = 0
        SENSOR_RANGE = consts.SENSOR_RANGE_M

        ######### get previous state & relevant metrics #########
        state = self.saved_state["state"]

        # avg distance from all sensors, either front, left, or right groupings
        # from [0, 1], 0 being no gap and 1 being massive gap
        front_gap = tools.avg(min(state[a] / SENSOR_RANGE, 1) for a in consts.SENSOR_ANGLES if abs(a) <= 10)
        left_gap  = tools.avg(min(state[a] / SENSOR_RANGE, 1) for a in consts.SENSOR_ANGLES if a < -10)
        right_gap = tools.avg(min(state[a] / SENSOR_RANGE, 1) for a in consts.SENSOR_ANGLES if a > 10)
        
        # scale the front gap by the following margin
        front_gap *= 1 - self.mood.margin
        left_gap = min(left_gap * SENSOR_RANGE / consts.TRACK_WIDTH_M / 2, 1)
        right_gap = min(right_gap * SENSOR_RANGE / consts.TRACK_WIDTH_M / 2, 1)

        # shifting changes
        has_accelerated = self.speed > state["speed"]
        has_upshifted = self.gear > state["gear"]
        has_downshifted = self.gear < state["gear"]

        ####################################
        ########## CRASH CHECKING ##########
        ####################################

        # check for collisions
        self._collision_check(track_poly, drivers)

        if self.has_crashed:
            # indicate to brake hard
            rewards[ACCEL] = 0
            rewards[BRAKE] = 1
            return rewards

        # check for money shifts
        if self.rpms >= consts.MAX_RPMS:
            # indicate to not shift
            self.has_crashed = True
            rewards[DOWNSHIFT] = 0
            return rewards

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

        ######### reward evaluation for state changes #########
        
        # evaluate accelerating and slowing near gaps
        min_gap = min(self.speed / (2 + 1.5 * self.mood.braking) / SENSOR_RANGE, 0.99) # in meters, around 2.75m space per 10m/s

        if front_gap < min_gap: # too close
            rewards[ACCEL] = 0
            rewards[BRAKE] = 1 - 0.8 * (front_gap / min_gap) # brake harder when gap is smaller
        else: # sufficient front gap
            scaled_gap = (front_gap - min_gap) / (1 - min_gap) # front gap starting from scaled_gap to max range
            rewards[ACCEL] = min(scaled_gap / (1 - self.mood.accel) + 0.5, 1) # accel harder when gap is larger
            rewards[BRAKE] = 0

        # handle lateral spacing
        rewards[LEFT] = 0.5 * (1 - left_gap) + 0.5 * right_gap
        rewards[RIGHT] = 0.5 * (1 - right_gap) + 0.5 * left_gap

        # reward and punish shift points
        shift_points = tools.get_shift_points(self.gear, self.mood.rev_bias)
        scaled_rpms = self.rpms / consts.MAX_RPMS # rpms from 0 to 1
        lower_pt = (shift_points[0] if shift_points[0] is not None else consts.IDLE_RPMS) / consts.MAX_RPMS
        upper_pt = shift_points[1] / consts.MAX_RPMS

        # 
        if has_upshifted:
            # reward based on how close the rpms are to the lower shift pt of the new gear
            rewards[UPSHIFT] = tools.clamp(0, 1, 1 - abs(scaled_rpms - lower_pt) / lower_pt)
        elif has_downshifted:
            # reward based on how close the rpms are to the upper shift pt of the new gear
            rewards[DOWNSHIFT] = tools.clamp(0, 1, 1 - abs(scaled_rpms - upper_pt) / upper_pt)
        else: # indicate whether or not to shift given the current rpms
            num_gears = len(consts.GEAR_RATIOS)

            can_moneyshift = tools.is_moneyshift_possible(self.speed, self.gear, self.mood.rev_bias)
            can_lug = tools.is_lugging_possible(self.speed, self.gear, self.mood.rev_bias)
            should_downshift = self.gear > 1 and self.rpms < shift_points[0]
            should_upshift = self.gear < num_gears and self.rpms > shift_points[1]

            # penalize potential money shifts and/or lugging
            if can_moneyshift or can_lug:
                if can_moneyshift: rewards[DOWNSHIFT] = 0
                if can_lug: rewards[UPSHIFT] = 0
            elif should_upshift or should_downshift: # reward based on shift points
                if should_downshift: # reward downshifts if lugging
                    rewards[DOWNSHIFT] = 1; rewards[UPSHIFT] = 0
                
                if should_upshift and has_accelerated: # reward for upshifting
                    rewards[DOWNSHIFT] = 0; rewards[UPSHIFT] = 1
                else: # punish down and upshifts when braking
                    rewards[DOWNSHIFT] = 0; rewards[UPSHIFT] = 0
            else: # no shifts are urgent--shift based on staying in lowest gear possible
                # aim for lowest gear possible (for accel: torque, for braking: engine braking; coasting not an issue)
                # given the speed, find the lowest gear possible
                target_gear = tools.get_target_gear(self.speed, self.mood.rev_bias)

                if target_gear < self.gear: # prioritize downshift
                    rewards[DOWNSHIFT] = 1; rewards[UPSHIFT] = 0
                elif target_gear > self.gear: # prioritize upshift
                    rewards[DOWNSHIFT] = 0; rewards[UPSHIFT] = 1
                else: # gear selection is good
                    rewards[DOWNSHIFT] = rewards[UPSHIFT] = 0

        return rewards

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
        if len(self.memory) == 0: return

        # extract data from bank
        inputs, outputs, rewards = [], [], []
        for memory in self.memory:
            inputs.append(memory["input"])
            outputs.append(memory["output"])
            rewards.append(memory["reward"])

        # calculate target data
        target_outputs = []
        
        int_factor = consts.INTERPOLATION_FACTOR
        for output, reward in zip(outputs, rewards):
            target_outputs.append([])
            for i in range(consts.NET_OUTPUT_SHAPE):
                target_outputs[-1].append( output[i] * (1 - int_factor) + reward[i] * int_factor )
    
        # compile training data
        train_x = np.array(inputs)[:,0,:].astype(np.float32)
        train_y = np.array(target_outputs).astype(np.float32)
        
        # train model
        self.model.fit(train_x, train_y, batch_size=consts.BATCH_SIZE, verbose=1,
                       validation_split=consts.VALIDATION_SPLIT, epochs=consts.TRAINING_EPOCHS)

        # wipe memory bank
        self.memory.clear()

    # export the model to a folder
    def export_model(self, folder: str) -> None:
        self.model.save(folder + "/" + str(self.car_num) + ".keras")

# #############################################
#
# TrainedDriver for loading existing models
#
# #############################################
class TrainedDriver(Driver):
    def __init__(self, model_src: str):
        super().__init__()
        self.model = tf.keras.models.load_model(model_src)