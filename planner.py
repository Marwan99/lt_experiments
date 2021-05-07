import cv2
import numpy as np

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


def plan_and_plot(start, end, finder, grid, colour):
    if not start.walkable:
        print("Start is occupied")
        output[start.y][start.x] = colour
        cv2.imshow('Map + path', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

    if not end.walkable:
        print("Goal is occupied")
        output[start.y][start.x] = colour
        cv2.imshow('Map + path', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

    path, runs = finder.find_path(start, end, grid)

    for pixel in path:
        output[pixel[1]][pixel[0]] = colour


# session = 'wed_teach'
session = 'repeat_0'
# session = 'repeat_2'
# session = 'repeat_4'
# session = 'repeat_6'

map = cv2.imread('/home/marwan/fyp/lt_experiments/d_2_n/' + session + '/map.pgm')

inverted_map = cv2.bitwise_not(map)

# Inflate by 3 cells, 3*0.15=0.45, robot raduis=~0.33
kernel = np.ones((6, 6), np.uint8)

inflated_map = cv2.dilate(inverted_map, kernel, iterations=1)
inflated_map = cv2.bitwise_not(inflated_map)

delta_zone = map - inflated_map
delta_zone[np.where((delta_zone == [254, 254, 254]).all(axis=2))] = [0, 0, 150]


# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = delta_zone.shape
roi = map[0:rows, 0:cols]

# Now create a mask of logo and create its inverse mask also
delta_gray = cv2.cvtColor(delta_zone, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(delta_gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(delta_zone, delta_zone, mask=mask)

output = cv2.add(img1_bg, img2_fg)
# cv2.imshow('Inflated map', output)


map_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
(thresh, bin_map) = cv2.threshold(map_gray,
                                  128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imshow('Binary map', bin_map)


# Path planning
grid = Grid(matrix=bin_map)
finder = AStarFinder(diagonal_movement=DiagonalMovement.always)

# Path 0
print("Path 0...")
start = grid.node(485, 335)
end = grid.node(336, 347)
plan_and_plot(start, end, finder, grid,[0, 150, 0])
print("Done")

# Path 1
print("Path 1...")
grid.cleanup()
start = grid.node(295, 400)
end = grid.node(450, 395)
plan_and_plot(start, end, finder, grid, [150, 0, 0])
print("Done")

cv2.imshow('Map + path', output)
cv2.imwrite('/home/marwan/fyp/lt_experiments/d_2_n/results/' + session + '_path.jpg', output)

cv2.waitKey(0)
cv2.destroyAllWindows()
