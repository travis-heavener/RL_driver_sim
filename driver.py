# hide Tensorflow welcome prompt (thanks https://stackoverflow.com/a/38645250)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from math import ceil
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, optimizers
import pygame

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
    follow      = 0.50  # higher values = longer following distance between drivers
    shifts      = 0.65  # higher values = shifting at higher rpms (avoid 0 and 1 lol)

    def __init__(self, accel: float=None, braking: float=None, speed: float=None,
                       follow: float=None, shifts: float=None):
        self.accel = accel or self.accel
        self.braking = braking or self.braking
        self.speed = speed or self.speed
        self.follow = follow or self.follow
        self.shifts = shifts or self.shifts

# preset DriverMoods
MOOD_NORMAL   = DriverMood() # uses defaults
MOOD_SPEEDY   = DriverMood(accel=0.95, braking=0.80, speed=0.99, follow=0.30, shifts=0.95)
MOOD_YOLO     = DriverMood(accel=1.00, braking=0.90, speed=1.00, follow=0.10, shifts=0.95)
MOOD_NERVOUS  = DriverMood(accel=0.50, braking=0.40, speed=0.70, follow=0.75, shifts=0.35)
MOOD_ECO      = DriverMood(accel=0.35, braking=0.70, speed=0.65, follow=0.50, shifts=0.30)

# #############################################
#
# Driver constructor for new, blank-slate driver models
#
# #############################################
class Driver:
    model: models.Sequential = None
    mood: DriverMood = MOOD_NORMAL
    img: pygame.Surface = None

    # physics
    x, y, width, length = 0, 0, 0, 0
    speed, accel = 0, 0 # linear, scalar

    # engine & powertrain
    direction: float = 0 # degrees
    throttle: float = 0
    gear: int = 1
    rpms: float = consts.IDLE_RPMS

    def __init__(self, mood: DriverMood=None):
        # copy image
        self.width = ceil(consts.VEHICLE_WIDTH_Y * consts.PX_YARD_RATIO)
        self.length = ceil(consts.VEHICLE_LENGTH_Y * consts.PX_YARD_RATIO)
        self.img = pygame.transform.scale( DRIVER_IMG, (self.length, self.width) )

        # create new blank-slate network
        self.model = models.Sequential()
        self.model.add(layers.Dense(24, input_shape=(consts.NET_INPUT_SHAPE,), activation="relu"))
        self.model.add(layers.Dense(48, activation='relu'))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(consts.NET_OUTPUT_SHAPE, activation="tanh"))

        # compile model
        self.model.compile(loss="mse", optimizer=optimizers.Adam(learning_rate=consts.LEARNING_RATE))

        # assign driver's mood
        self.mood = mood or self.mood
    
    def set_pos(self, x: float, y: float) -> None:
        self.x = x; self.y = y

    #
    # enables the driver to make a decision, evaluate it, and update the model if they haven't crashed
    #
    def update(self, drivers: list[any], dt: float) -> None:
        pass
        # self.direction += 360 * dt
        # self.direction %= 360
        # self.x += 4 * consts.PX_YARD_RATIO  * dt

    #
    # draws the driver on the window
    #
    def draw(self, window: pygame.Surface) -> None:
        img = pygame.transform.rotate(self.img, self.direction)
        center_pos = (self.x - self.length / 2, self.y - self.width / 2) # shifted back half-way to draw on middle
        center_rect = self.img.get_rect(topleft=center_pos).center
        pos = img.get_rect(center=center_rect)

        window.blit(img, pos)

# #############################################
#
# TrainedDriver for loading existing models
#
# #############################################
class TrainedDriver(Driver):
    pass