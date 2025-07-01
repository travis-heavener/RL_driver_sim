# Hide pygame welcome message (thanks https://stackoverflow.com/a/55769463)
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import json

from driver import *
from track import Track
from sim import SimContainer

#
# Main function
#
def main(config: dict):
    # Load track from disc
    track = Track(config["track"])

    # Create drivers
    drivers: list[Driver] = []
    if len(config["racers"]) == 0:
        drivers.extend([ Driver() ])
    else:
        for driver in config["racers"]:
            if driver["model"]["loadFrom"] is None:
                drivers.append(Driver(
                    mood=MOODS[ driver["mood"] ],
                    pos_restore_status=config["config"]["resetAfterTraining"],
                    model_save_path=driver["model"]["saveTo"]
                ))
            else:
                drivers.append(TrainedDriver(
                    driver["model"]["loadFrom"],
                    mood=MOODS[ driver["mood"] ],
                    pos_restore_status=config["config"]["resetAfterTraining"],
                    model_save_path=driver["model"]["saveTo"]
                ))

    # Configure training epsilon
    if config["config"]["useZeroEpsilon"]:
        for driver in drivers:
            driver.set_epsilon(0)

    # Configure sim constants
    consts.set_use_antialias( config["config"]["useAntialiasing"] )

    # Start the simulation
    sim = SimContainer(track) # Initialize simulation window
    sim.addDrivers(*drivers) # Add drivers to sim
    sim.run(config)

if __name__ == "__main__":
    # Load config
    config = None
    with open("setup.json", "r") as f:
        config = json.load(f)
    main(config)