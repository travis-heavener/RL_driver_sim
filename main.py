# hide pygame welcome message (thanks https://stackoverflow.com/a/55769463)
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
from sys import argv

from driver import *
from track import Track
from sim import SimContainer

#
# main function
#
def main(track_src: str, kwargs: dict, model_srcs=None):
    # load track from disc
    track = Track(consts.TRACKS_FOLDER + track_src + ".csv")

    # create drivers
    drivers = []
    if model_srcs is None:
        drivers.extend([
            # Driver(), Driver(MOOD_SPEEDY), Driver(MOOD_YOLO), Driver(MOOD_NERVOUS), Driver(MOOD_ECO)
            Driver()
        ])
    else:
        drivers.extend([ TrainedDriver(consts.MODELS_FOLDER + src + ".keras") for src in model_srcs ])

    if kwargs["no_pos_restore"]:
        for driver in drivers:
            driver.set_pos_restore(False)
    if kwargs["zero_epsilon"]:
        for driver in drivers:
            driver.set_epsilon(0)

    sim = SimContainer(track) # initialize simulation window
    sim.addDrivers(*drivers) # add drivers to sim

    sim.run(kwargs) # start the simulation

if __name__ == "__main__":
    if len(argv) < 2:
        print("Err: Missing track source file argument.")
        exit(1)
    
    track_src = argv[1]
    kwargs = {
        "show_bboxes": "show_bboxes" in argv,
        "show_sensors": "show_sensors" in argv,
        "show_driveline": "show_driveline" in argv,
        "will_save_models": "no_save" not in argv,
        "no_pos_restore": "no_restore" in argv,
        "will_log": "show_logs" in argv,
        "zero_epsilon": "zero_epsilon" in argv # eliminate random desicion making
    }

    model_srcs = None
    for arg in argv:
        if arg.startswith("models="):
            model_srcs = arg.split("=", 1)[1].split(",")
            model_srcs = [ src.strip() for src in model_srcs ]
            break

    # use first argument as source file
    main(track_src, kwargs, model_srcs)