import cv2
import numpy as np
import random

SCALE_FACTOR = 1 #Conversion from pixels to unit of choice


def morphImage(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((6, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=6)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 3050

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2

def draw_ellipses(image):
    image = np.uint8(image)

    if len(image.shape) == 3:
        b_w = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("happens")
    else:
        b_w = image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


    print(image.dtype)

    contours, _ = cv2.findContours(b_w, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)



    # Find the rotated rectangles and ellipses for each contour
    minEllipse = [None] * len(contours)
    for i, c in enumerate(contours):
          if c.shape[0] > 5:
              minEllipse[i] = cv2.fitEllipse(c)
    # Draw contours + rotated rects + ellipses

    #drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i, c in enumerate(contours):
        color = (0, 0, 255)
        # contour
        cv2.drawContours(image, contours, i, color)
        # ellipse
        if c.shape[0] > 5:
            cv2.ellipse(image, minEllipse[i], color, 2)
        # rotated rectangle

    return (image, minEllipse)

def calc_averages(ellipseData):
    area = 0
    maj_over_min = 0
    for data_group in ellipseData:
        maj_axis, min_axis = data_group[1]
        area += maj_axis * min_axis * 3.1415 * SCALE_FACTOR * SCALE_FACTOR
        maj_over_min += maj_axis/min_axis
    return (area/(len(ellipseData)/3), maj_over_min/(len(ellipseData)/3))



t2 = cv2.imread('/Users/alex/Downloads/I-tran-dry1-threshold-test.tif')

t2 = morphImage(t2)

t2, ellipseData = draw_ellipses(t2)

print(ellipseData)

copy = cv2.resize(t2, (0,0), fx=0.3, fy=0.3)

cv2.imshow("thing", copy)

area_average, maj_over_min_average = calc_averages(ellipseData)

print("Average Area: ", area_average, "Average Major Axis / Minor Axis:", maj_over_min_average)

cv2.imwrite("/Users/alex/Downloads/I-tran-dry1-ellipse.png", t2)

cv2.waitKey()

