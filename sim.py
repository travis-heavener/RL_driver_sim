# fix Windows DPI scaling (thanks https://stackoverflow.com/a/32063729)
import ctypes
ctypes.windll.user32.SetProcessDPIAware()

import pygame
import pygame.freetype
from time import time

import consts
from consts import WIDTH, HEIGHT, FPS, GRASS_COLOR_RGB
from driver import Driver
from track import Track

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

    def run(self, config: dict) -> None:
        is_running = True
        last_trained_ts = time()
        last_frame_rate = last_frame_rate_ts = 0
        generation_num = 1

        # Game loop
        while is_running:
            # Check for closure request event (thanks https://www.pygame.org/docs/)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

            # Update drivers
            start = time()
            if self.dt > 0 and self.dt < 0.15: # Prevent large frame skips from affecting movement
                for driver in self.drivers: # Move drivers first
                    driver.move(self.track.track_poly, self.drivers, self.dt)

                # Rate each driver's move
                for driver in self.drivers:
                    driver.evaluate(self.track.track_poly, self.drivers)
                    if config["config"]["logTelemetry"]:
                        driver.log()

            # Render frame
            self.window.fill(GRASS_COLOR_RGB) # Wipe screen
            self.track.draw(self.window, show_driveline=config["config"]["showDriveline"]) # Render track

            # Render each driver
            for driver in self.drivers:
                driver.draw(
                    self.window,
                    draw_bbox=config["config"]["showHitboxes"],
                    draw_sensor_paths=config["config"]["showSensors"]
                )

            # Display FPS
            if time() - last_frame_rate_ts >= consts.FPS_DISPLAY_RATE:
                last_frame_rate_ts = time()
                frame_time_s = last_frame_rate_ts - start
                last_frame_rate = min(round(1 / max(frame_time_s, 1e-5)), FPS) # Prevent zero div

            text_surface, text_rect = self.font_medium.render(f"{last_frame_rate} FPS", consts.TEXT_COLOR_RGB)
            self.window.blit(text_surface, (WIDTH - (HEIGHT // 100) - text_rect.width, HEIGHT // 100))

            # Display generation readout
            text_surface, text_rect = self.font_medium.render(f"Generation #{generation_num}", consts.TEXT_COLOR_RGB)
            self.window.blit(text_surface, (HEIGHT // 100, HEIGHT // 100))

            # Display frame
            pygame.display.flip()

            # Check if the generation has ended
            is_driver_remaining = False
            for driver in self.drivers:
                if not driver.has_crashed:
                    is_driver_remaining = True
                    break
            
            has_trained = False
            if not is_driver_remaining or time() - last_trained_ts > consts.MAX_GENERATION_TIME:
                generation_num += 1

                # Train models
                for driver in self.drivers:
                    driver.train()
                    driver.reset()

                has_trained = True
                last_trained_ts = time() # Update last training timestamp

            if generation_num > consts.NUM_GENERATIONS:
                is_running = False

            # Extract frame time gap
            if has_trained: # Ignore frametime if training (causes REALLY large dt values and can jump the track)
                self.dt = 0
            else:
                self.dt = self.clock.tick(FPS) / 1e3

        # End game
        pygame.quit()

        # Export models
        for driver in self.drivers:
            driver.export_model() # Export model