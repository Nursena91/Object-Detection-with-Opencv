import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
general = cv.imread('game2.png', cv.IMREAD_UNCHANGED)
white_circle = cv.imread('white_circle.png', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(general, white_circle, cv.TM_CCOEFF_NORMED)
plt.imshow(result)
plt.show()
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
threshold = 0.5
print(max_val)
if max_val >=threshold:
    circle_w = white_circle.shape[1]
    circle_h = white_circle.shape[0]
    top_left = max_loc
    bottom_right = (top_left[0] + circle_w, top_left[1] + circle_h)
    cv.rectangle(general, top_left, bottom_right,
                 color=(0,255,0), thickness=2,lineType=cv.LINE_4)

else:
    print("Couldn't find white circle")


black_circle = cv.imread('black_circle.png', cv.IMREAD_UNCHANGED)

result2 = cv.matchTemplate(general, black_circle, cv.TM_CCOEFF_NORMED)

min_val2, max_val2, min_loc2, max_loc2 = cv.minMaxLoc(result2)
threshold2 = 0.4
print(max_val2)
if max_val2 >=threshold2:
    circle_w2 = black_circle.shape[1]
    circle_h2 = black_circle.shape[0]
    top_left2 = max_loc2
    bottom_right2 = (top_left2[0] + circle_w2, top_left2[1] + circle_h2)
    cv.rectangle(general, top_left2, bottom_right2,
                 color=(0,255,0), thickness=2,lineType=cv.LINE_4)
    general_rgb = cv.cvtColor(general, cv.COLOR_BGR2RGB)
    plt.imshow(general_rgb)
    plt.show()
else:
    print("Couldn't find black circle")
"""

def findClickPositions(object_image_path, general_image_path, threshold=0.6,debug_mode=None ):

    general = cv.imread(general_image_path, cv.IMREAD_UNCHANGED)
    object = cv.imread(object_image_path, cv.IMREAD_UNCHANGED)

    object_w = object.shape[1]
    object_h = object.shape[0]

    method = cv.TM_CCOEFF_NORMED
    result = cv.matchTemplate(general, object, method)

    locations = np.where(result>=threshold)
    locations = list(zip(*locations[::-1]))

    rectangles = []
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), object_w, object_h]
        rectangles.append(rect)
        


    rectangles, weights = cv.groupRectangles(rectangles,1,0.9)
    #print(rectangles)

    points = []
    if len(rectangles):
        line_color = (0,255,0)
        line_type = cv.LINE_4
        marker_color = (0,255,0)
        marker_type = cv.MARKER_CROSS
    
        for (x,y,w,h) in rectangles:

            center_x = x + int(w/2)
            center_y = y + int(h/2)
            points.append((center_x, center_y))
            if debug_mode == 'rectangles':
                top_left = (x,y)
                bottom_right = (x + w, y + h)
                cv.rectangle(general, top_left, bottom_right, line_color, line_type)
            elif debug_mode == 'points':
                cv.drawMarker(general, (center_x,center_y), marker_color, marker_type)

        if debug_mode: 
            general_rgb = cv.cvtColor(general, cv.COLOR_BGR2RGB)
            plt.imshow(general_rgb)
            plt.show()

    return points

points = findClickPositions('square.png', 'game.png',threshold=0.5, debug_mode = 'points')
print(points)

# kare için = 0.6
# siyah daire için = 0.45 0.5 arası (0.4 de olabilir) 
# beyaz daire için 0.5


