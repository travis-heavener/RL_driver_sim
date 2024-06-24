from math import floor
import numpy as np
import pygame

import consts
import tools

"""

    TRACK FILE FORMAT:
        LINES 1-12:
            CAR STARTING GRID POSITIONS, FACING RIGHT
        LINES 13+:
            TRACK VERTICES, USE YARDS AS UNITS FOR BEST PRACTICE

"""

class Track:
    # the vertices for pivots in the track's geometry, connected by spline curves
    # thanks https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
    vertices: np.ndarray
    grid: np.ndarray # starting positions
    start_line: np.ndarray
    inner_wall: np.ndarray
    outer_wall: np.ndarray
    track_poly: np.ndarray

    def __init__(self, track_src: str):
        MAX_WIDTH = consts.TRACK_BOUNDS[1][0] - consts.TRACK_BOUNDS[0][0]
        MAX_HEIGHT = consts.TRACK_BOUNDS[1][1] - consts.TRACK_BOUNDS[0][1]

        # load track from file
        vertices: list[tuple[int, int]] = []
        grid: list[tuple[int, int]] = []
        line: str = None
        with open(track_src) as handle:
            # read first 12 lines for starting grid
            i = 0
            while line := handle.readline():
                i += 1

                # skip invalid lines
                if not line.find(",") or i == 13: continue

                line = line.split(",")
                pos = (float(line[0]), -float(line[1]))
                if i <= 12: # extract starting position
                    grid.append(pos)
                else: # extract vertices
                    vertices.append(pos)

        if vertices[0] != vertices[-1]:
            vertices.append(vertices[0]) # add last vertex again to close the shape
        self.vertices = np.array(vertices)
        self.grid = np.array(grid)
        
        # generate outer spline curve from vertices
        tck, outer_wall = tools.gen_outer_spline(np.array(vertices))

        # calculate centroid w/o extra last coordinate
        c_x, c_y = np.mean(outer_wall[:-1], axis=0)
        outer_wall -= np.array([c_x, c_y]).T # center at origin
        self.grid -= np.array([c_x, c_y]).T # adjust starting grid to origin

        # scale both walls together
        c_track = ( # bottom-left + midpoint
            consts.TRACK_BOUNDS[0][0] + MAX_WIDTH / 2,
            consts.TRACK_BOUNDS[0][1] + MAX_HEIGHT / 2
        )
        scale_factor = min( (MAX_WIDTH, MAX_HEIGHT) / np.ptp(outer_wall, axis=0) )

        # update pixels per yard ratio
        consts.set_px_yard_ratio(scale_factor)

        # scale and center on frame
        outer_wall *= scale_factor
        self.grid *= scale_factor
        outer_wall += np.array(c_track).T
        self.grid += np.array(c_track).T

        # generate inner wall
        inner_wall = tools.gen_inner_spline(tck, outer_wall)

        # create polygon from track walls (flip inner wall to keep continuity)
        self.inner_wall = inner_wall.astype(int)
        self.outer_wall = outer_wall.astype(int)
        self.track_poly = np.vstack((self.outer_wall, self.inner_wall[::-1])).astype(int)
        self.start_line = np.array([outer_wall[0], inner_wall[0]]).astype(int)

        # define draw constants
        self.__START_PX = round(consts.START_WIDTH_Y * consts.PX_YARD_RATIO)
        self.__BARRIER_PX = round(consts.BARRIER_WIDTH_Y * consts.PX_YARD_RATIO)

    def draw(self, window: pygame.Surface) -> None:
        # render track itself
        pygame.draw.polygon(window, consts.TRACK_COLOR_RGB, self.track_poly)

        # render start line
        pygame.draw.line(window, consts.FINISH_COLOR_RGB, *self.start_line, self.__START_PX)

        # render track walls onto frame
        pygame.draw.lines(window, consts.BARRIER_COLOR_RGB, False, self.outer_wall, self.__BARRIER_PX)
        pygame.draw.lines(window, consts.BARRIER_COLOR_RGB, False, self.inner_wall, self.__BARRIER_PX)