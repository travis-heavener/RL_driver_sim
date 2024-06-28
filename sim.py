# hide pygame welcome message (thanks https://stackoverflow.com/a/55769463)
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# fix Windows DPI scaling (thanks https://stackoverflow.com/a/32063729)
import ctypes
ctypes.windll.user32.SetProcessDPIAware()

import numpy as np
import pygame
from time import time

import consts
from consts import WIDTH, HEIGHT, FPS, GRASS_COLOR_RGB
from driver import Driver, TrainedDriver
from track import Track
import tools

#
# Holds all code pertaining to rendering the window and
# executing functions during game ticks.
#

class SimContainer:
    # references to Pygame objects
    window: pygame.Surface = None
    clock = None

    # reference to Track
    track: Track = None

    dt = 0 # delta time elapsed between render frames

    # references to all drivers
    drivers: list[Driver] = []
    
    def __init__(self, track: Track):
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.track = track

        pygame.display.set_caption("RLDS: Circuit Edition")
    
    def addDrivers(self, *drivers: list[Driver]) -> None:
        # put drivers into starting grid
        i = len(self.drivers)

        for driver in drivers:
            driver.set_start_pos(*self.track.grid[i])
            i += 1

        self.drivers.extend(drivers)

    def run(self) -> None:
        is_running = True
        last_trained_ts = time()
        generation_num = 1

        # game loop
        while is_running:
            # check for closure request event (thanks https://www.pygame.org/docs/)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
            
            # update drivers
            start = time()
            if self.dt > 0:
                for driver in self.drivers: # move drivers first
                    driver.move(self.track.track_poly, self.drivers, self.dt)

                for driver in self.drivers: # rate each driver's move
                    driver.evaluate(self.track.track_poly, self.drivers)
            print(f"Draw time: {round(time() - start, 5)}s")

            # render frame
            self.window.fill(GRASS_COLOR_RGB) # wipe screen
            self.track.draw(self.window, show_driveline=True) # render track

            for driver in self.drivers: # render each driver
                driver.draw(self.window, draw_bbox=False)

            # display frame
            pygame.display.flip()

            # check if the generation has ended
            is_driver_remaining = False
            for driver in self.drivers:
                if not driver.has_crashed:
                    is_driver_remaining = True
                    break
            
            if not is_driver_remaining or time() - last_trained_ts > consts.MAX_GENERATION_TIME:
                tools.log(f"Generation #{generation_num} ended.")
                generation_num += 1

                # train models
                for driver in self.drivers:
                    driver.train()
                    driver.reset()

                last_trained_ts = time() # update last training timestamp
                tools.log("Training complete.")

            if generation_num > consts.NUM_GENERATIONS:
                is_running = False

            # extract frame time gap
            self.dt = self.clock.tick(FPS) / 1e3

        # end game
        pygame.quit()

        # export models
        print("TODO: export models & display summary")