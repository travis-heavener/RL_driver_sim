# hide pygame welcome message (thanks https://stackoverflow.com/a/55769463)
import os

import pygame.freetype
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
    font: pygame.freetype.Font = None

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

        # load fonts
        self.font_regular = pygame.freetype.Font("res/font_regular.ttf", consts.FONT_SIZE)
        self.font_medium = pygame.freetype.Font("res/font_medium.ttf", consts.FONT_SIZE)
        self.font_bold = pygame.freetype.Font("res/font_bold.ttf", consts.FONT_SIZE)

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
        last_frame_rate = last_frame_rate_ts = 0
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

            # render frame
            self.window.fill(GRASS_COLOR_RGB) # wipe screen
            self.track.draw(self.window, show_driveline=True) # render track

            for driver in self.drivers: # render each driver
                driver.draw(self.window, draw_bbox=False)

            # display FPS
            if time() - last_frame_rate_ts >= consts.FPS_DISPLAY_RATE:
                last_frame_rate_ts = time()
                frame_time_s = last_frame_rate_ts - start
                last_frame_rate = min(round(1 / max(frame_time_s, 1e-5)), FPS) # prevent zero div
                
            text_surface, text_rect = self.font_medium.render(f"{last_frame_rate} FPS", consts.TEXT_COLOR_RGB)
            self.window.blit(text_surface, (WIDTH - (HEIGHT // 100) - text_rect.width, HEIGHT // 100))

            # display generation readout
            text_surface, text_rect = self.font_medium.render(f"Generation #{generation_num}", consts.TEXT_COLOR_RGB)
            self.window.blit(text_surface, (HEIGHT // 100, HEIGHT // 100))


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