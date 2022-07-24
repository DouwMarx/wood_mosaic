import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_rotated(contour, image, name,fixed_sidelength=False):
    mult = 1  # Show an area slightly larger than the rectange. Set to 1 in production

    rect = cv2.minAreaRect(contour)  # Get minimum rectange of image
    box = cv2.boxPoints(rect)  # Corner coordinates
    box = np.int0(box)  # Convert to index / integer

    W = rect[1][0] # Rectangle width
    H = rect[1][1] # Rectangle height

    Xs = [i[0] for i in box] # x coordinates
    Ys = [i[1] for i in box] # y coordinates

    # Extent of the rectangle
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    # Make sure to work with a positive angle
    rotated = False
    angle = rect[2]
    if angle < -45:
        angle += 90
        rotated = True

    # Compute the center of the rectange
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    sidelenghts = [W,H] # Compute sidelengths for later use

    # Compute the size of the rotated object (height and width)
    size = (int(mult * (x2 - x1)), int(mult * (y2 - y1)))

    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0) # For rotating image

    # Crop and rotate
    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)


    if fixed_sidelength is False:
        croppedW = W if not rotated else H
        croppedH = H if not rotated else W
    else:
        croppedW = int(fixed_sidelength)
        croppedH = int(fixed_sidelength)

    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW * mult), int(croppedH * mult)),
                                       (size[0] / 2, size[1] / 2))

    cv2.imwrite("cropped_and_rotated" + name + ".jpeg", croppedRotated)
    return sidelenghts
