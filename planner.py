import cv2
import numpy as np
from operator import itemgetter
from operator import attrgetter
import math


def heuristic(node, goal):
    D = 1
    D2 = math.sqrt(2)
    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end, explored):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Initialize both open and closed list
    open_list = [start_node]
    closed_list = []

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the node with min f value
        open_list.sort(key=lambda x: (x.f, x.g))
        current_node = open_list[0]

        # Pop current off open list, add to closed list
        open_list.pop(0)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Adjacent squares
        for neighbour in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            # for neighbour in [(0, -1), (0, 1), (-1, 0), (1, 0)]:

            # Get node position
            node_position = (
                current_node.position[0] + neighbour[0],
                current_node.position[1] + neighbour[1])

            # Make sure within range
            row, col = maze.shape
            if node_position[1] > (col-1) or node_position[1] < 0 or node_position[0] < 0 or node_position[0] > (row - 1):
                continue

            # if pixel is not white aka occupied
            if maze[node_position[0]][node_position[1]] != 255:
                continue

            new_node = Node(current_node, node_position)

            # Check if closed
            closed = False
            for closed_node in closed_list:
                if new_node == closed_node:
                    closed = True
            if closed:
                continue

            # Calculate the f, g, and h values
            new_node.g = current_node.g + 1
            # new_node.h = math.sqrt(((new_node.position[0] - end_node.position[0]) **
            #                         2) + ((new_node.position[1] - end_node.position[1]) ** 2))
            new_node.h = heuristic(new_node.position, end_node.position)
            new_node.f = new_node.g + new_node.h

            # Child is already in the open list
            opened = False
            for index, open_node in enumerate(open_list):
                if new_node == open_node:
                    opened = True
                    if (new_node.g < open_node.g):
                        open_list[index] = new_node

            if not opened:
                open_list.append(new_node)

            explored[node_position[0]][node_position[1]] = new_node.g+10
            cv2.imshow('exploring map', explored)

        # cv2.waitKey(0)


map = cv2.imread('/home/marwan/fyp/lt_experiments/d_2_n/wed_teach/map.pgm')

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

# row, col
# y, x

start = (290, 415)
# start = (363, 397)
goal = (360, 235)

if bin_map[start[0]][start[1]] == 0:
    print("Start is occupied")
    exit()

if bin_map[goal[0]][goal[1]] == 0:
    print("Goal is occupied")
    exit()

output[start[0]][start[1]] = [0, 150, 0]
output[goal[0]][goal[1]] = [0, 150, 0]
# cv2.imshow('Map + path', output)

path = astar(bin_map, start, goal, bin_map)

for pixel in path:
    output[pixel[0]][pixel[1]] = [0, 150, 0]


cv2.imshow('Map + path', output)


cv2.waitKey(0)
cv2.destroyAllWindows()
