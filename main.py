from sys import argv

from driver import *
from track import Track
from sim import SimContainer

#
# main function
#
def main(track_src: str):
    # load track from disc
    track = Track("tracks/" + track_src + ".csv")

    # create drivers
    drivers = [
        # Driver(), Driver(MOOD_SPEEDY), Driver(MOOD_YOLO), Driver(MOOD_NERVOUS), Driver(MOOD_ECO)
        Driver(),
        Driver(),
        Driver(),
        Driver(),
        Driver(),
        Driver(),
        Driver(),
        Driver(),
        Driver(),
        Driver(),
        Driver(),
        Driver()
    ]

    sim = SimContainer(track) # initialize simulation window
    sim.addDrivers(*drivers) # add drivers to sim

    sim.start() # start the simulation

    # NOTE: code executed here is after game completion

if __name__ == "__main__":
    if len(argv) < 2:
        print("Err: Missing track source file argument.")
        exit(1)
    
    # use first argument as source file
    main(argv[1])