import cv2
import numpy as np
import sys
import math
from collections import deque
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pprint
import copy

sys.setrecursionlimit(10000)

# Read the original image
img = cv2.imread('test.png')
# Display original image
# cv2.imshow('Original', img)
# cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

# with np.printoptions(threshold=np.inf):
#     print(edges)

# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

# cv2.destroyAllWindows()

# Below lists detail all eight possible movements from a cell
# (top, right, bottom, left, and four diagonal moves)
row = [-1, -1, -1, 0, 1, 0, 1, 1]
col = [-1, 1, 0, -1, -1, 1, 0, 1]


# Function to check if it is safe to go to position (x, y)
# from the current position. The function returns false if (x, y)
# is not valid matrix coordinates or (x, y) represents water or
# position (x, y) is already processed.

def isSafe(mat, x, y, processed):
    return (x >= 0 and x < len(processed)) and (y >= 0 and y < len(processed[0])) and \
           mat[x][y] == 255 and not processed[x][y]


def BFS(mat, processed, i, j):
    # create an empty queue and enqueue source node
    q = deque()
    q.append((i, j))

    # mark source node as processed
    processed[i][j] = True
    thisIsland = []
    stepsInSearchDirection = 25

    # loop till queue is empty
    while q:
        # dequeue front node and process it
        x, y = q.popleft()

        # check for all eight possible movements from the current cell
        # and enqueue each valid movement
        for k in range(len(row)):
            for step in range(1, stepsInSearchDirection): # steps in any direction
                rowIndex = row[k] * step
                colIndex = col[k] * step

                # skip if the location is invalid, or already processed, or has water
                if isSafe(mat, x + rowIndex, y + colIndex, processed):
                    # skip if the location is invalid, or it is already
                    # processed, or consists of water
                    processed[x + rowIndex][y + colIndex] = True
                    thisIsland.append((x + rowIndex, y + colIndex))
                    q.append((x + rowIndex, y + colIndex))

    return thisIsland

def getAllIslands(mat):
    # base case
    if not mat.any() or not len(mat):
        return 0

    # `M × N` matrix
    (M, N) = (len(mat), len(mat[0]))

    # stores if a cell is processed or not
    processed = [[False for x in range(N)] for y in range(M)]

    allIslands = []

    island = 0
    for i in range(M):
        for j in range(N):
            # start BFS from each unprocessed node and increment island count
            if mat[i][j] == 255 and not processed[i][j]:
                thisIsland = BFS(mat, processed, i, j)
                allIslands.append(thisIsland)
                island = island + 1

    print(allIslands)

    return allIslands

allIslands = getAllIslands(edges)

allIslands.sort(key=len)

allIslands = allIslands[:-1]


print("number of islands", len(allIslands))

def tanFunctionTransform(i, j):
    numSteps = 100
    return (i, j + numSteps)

def checkIfInBounds(island, i, j):
    # return (i,j) in island

    x,y = zip(*island)
    minX = min(x)
    minY = min(y)
    maxX = max(x)
    maxY = max(y)

    if (i > minX and i < maxX and j > minY and j < maxY):
        print(i, j, "is in:", len(island))
        return True

    return False

# edges = [
#     [255, 255, 255, 0, 0, 0, 255],
#     [0, 255, 255, 0, 0, 0, 255],
#     [0, 0, 0, 0, 0, 0, 255],
#     [255, 255, 255, 0, 0, 0, 255],
#     [0, 0, 0, 0, 0, 0, 255],
#     [255, 255, 255, 0, 0, 0, 255],
# ]

# allIslands = [
#     [(0,0), (0,1), (0,2), (1,1), (1,2)],
#     [(0,6), (1,6), (2,6), (3,6), (4,6), (5,6)],
#     [(3,0), (3,1), (3,2)],
#     [(5,0), (5,1), (5,2)],
# ]



newimg = copy.copy(img)

for i in range(newimg.shape[0]):
    for j in range(newimg.shape[1]):
        for island in allIslands:
            if checkIfInBounds(island, i, j):
                transformedCoord = tanFunctionTransform(i, j)
                if 0 < transformedCoord[0] < newimg.shape[0] and 0 < transformedCoord[1] < newimg.shape[1]:
                    x = transformedCoord[0]
                    y = transformedCoord[1]

                    tempxy = copy.copy(newimg[x][y])
                    tempij = copy.copy(newimg[i][j])

                    newimg[i][j] = tempxy
                    newimg[x][y] = tempij

                    print("newimg[x][y]", x, y, i, j, newimg[i][j], newimg[x][y])
                else:
                    newimg[i][j] = [0, 0, 0]



cv2.imshow('Transformed image', newimg)
cv2.waitKey(0)

#save matrix/array as image file
isWritten = cv2.imwrite('./test-update.png', newimg)

if isWritten:
	print('Image is successfully saved as file.')