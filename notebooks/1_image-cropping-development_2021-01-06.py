import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from src.utils import show_large_image
from src.cropping import crop_rotated
from joblib import Parallel, delayed
import itertools

def do_cropping(filename,write_intermediate=True):
    f = os.path.join(directory, filename)
    image = cv2.imread(f)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if write_intermediate:
        cv2.imwrite("gray_" + filename + ".jpeg", gray)

    # Median blur to smooth the image. This removes the noise in the image. The central element is replaced by the
    # median of all pixels in a kernel area
    kernel_size_medblur = 21
    blur = cv2.medianBlur(gray, kernel_size_medblur)
    if write_intermediate:
        cv2.imwrite("blur_" + filename + ".jpeg", blur)

    # Shapening Kernel for pronounced edges
    sharpen_kernel = np.array([[-1,-1,-1],
                               [-1,9,-1],
                               [-1,-1,-1]]
                              )

    ddepth = -1 # Output image will have the same depth as the source.
    # Depth is the number of bits used to represent color in the image
    sharpen = cv2.filter2D(blur, ddepth, sharpen_kernel)
    if write_intermediate:
        cv2.imwrite("sharpen_" + filename + ".jpeg", sharpen)

    # Threshold to convert the image to a binary image.
    # Set threshold dynamically based on max pixel intensity
    threshold = 0.25*np.max(sharpen) # Everything above this is 1, otherwise 0 (could be other way around?)
    # Lowering threashold incorporates more light colored areas
    thresh = cv2.threshold(sharpen,threshold,255, cv2.THRESH_BINARY_INV)[1]
    if write_intermediate:
        cv2.imwrite("threshold_" + filename + ".jpeg", thresh)

    # Perform morphological operations:
    morph_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size,morph_size))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=8)

    # Get the contours
    cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] # Get the contours and not the hierarchy

    image_area = gray.shape[0]*gray.shape[1] # Estimate the area of the contour
    # number_of_squares = 1
    # guessed_fraction_of_photo_occupied_by_squares = 0.25
    # expected_block_area = image_area*guessed_fraction_of_photo_occupied_by_squares/number_of_squares

    expected_block_area = 833204 # Area of a block based on previous iterations

    min_area = expected_block_area*0.9
    max_area = expected_block_area*1.1

    side_lengths =[] # Extract the sidelengths from the images so that they can be used in future as a hardcoded parameter
    cropped = False
    cropped_area = np.nan
    # Loop through all extracted contours and choose one that corresponds with the expected area of the contour
    for i,contour in enumerate(cnts):
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            print("Selected Area between min and max:",min_area, area,max_area)
            side_lengths+=crop_rotated(contour, image,filename,fixed_sidelength = 921) # Do cropping of picture. Sidelength hard coded based on median side length.
            cropped = True
            cropped_area = area
            break

    if cropped is not True:
        print("Cropping unsuccessful for:", filename)

    return cropped_area,side_lengths

directory = '../data/sample_set'
r = Parallel(n_jobs=6)(delayed(do_cropping)(fname) for fname in os.listdir(directory))
r = [list(i) for i in zip(*r)]
areas = r[0]
side_lengths = r[1]
side_lengths = list(itertools.chain(*side_lengths))

print(side_lengths)
print("Average area: ", np.average(areas))
print("Median side length: ", np.median(side_lengths))
