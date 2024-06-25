# hide pygame welcome message (thanks https://stackoverflow.com/a/55769463)
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# fix Windows DPI scaling (thanks https://stackoverflow.com/a/32063729)
import ctypes
ctypes.windll.user32.SetProcessDPIAware()

import pygame

from consts import WIDTH, HEIGHT, FPS, GRASS_COLOR_RGB
from driver import Driver, TrainedDriver
from track import Track

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
            driver.set_pos(*self.track.grid[i])
            i += 1

        self.drivers.extend(drivers)

    def run(self) -> None:
        running = True

        # game loop
        while running:
            # check for closure request event (thanks https://www.pygame.org/docs/)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # move drivers
            if self.dt > 0:
                for driver in self.drivers:
                    driver.update(self.track.track_poly, self.drivers, self.dt)

            # render frame
            self.window.fill(GRASS_COLOR_RGB) # wipe screen
            self.track.draw(self.window) # render track

            for driver in self.drivers: # render each driver
                driver.draw(self.window)

            pygame.draw.line(self.window, (255,0,0), (999,846), (1015,834), 4)

            pygame.display.flip() # display frame
            
            # extract frame time gap
            self.dt = self.clock.tick(FPS) / 1e3

        # end game
        pygame.quit()

        # export models
        print("TODO: export models & display summary")