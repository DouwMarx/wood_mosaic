import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from src.utils import show_large_image
from src.cropping import crop_rotated
from joblib import Parallel, delayed
import itertools
import pathlib

# source_directory = pathlib.Path('../data/sample_set')
# target_directory = pathlib.Path('../data')

source_directory = pathlib.Path('/mnt/EE30E36B30E338EB/Data/wood_mnist/wood_mnist/problem_files')
# source_directory = pathlib.Path('/mnt/EE30E36B30E338EB/Data/wood_mnist/wood_mnist/images')
target_directory = pathlib.Path('/mnt/EE30E36B30E338EB/Data/wood_mnist/wood_mnist/processed')

def do_cropping(filename,source_directory, target_directory,write_intermediate=True):

    # intermediate_dir =
    # out_dir

    f = os.path.join(source_directory, filename)
    image = cv2.imread(f)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if write_intermediate:
        cv2.imwrite(str(target_directory.joinpath("intermediate","gray_" + filename + ".jpeg")), gray)

    # Median blur to smooth the image. This removes the noise in the image. The central element is replaced by the
    # median of all pixels in a kernel area
    kernel_size_medblur = 61
    # kernel_size_medblur = 101
    blur = cv2.medianBlur(gray, kernel_size_medblur)
    if write_intermediate:
        cv2.imwrite(str(target_directory.joinpath("intermediate","blur" + filename + ".jpeg")), blur)

    # Shapening Kernel for pronounced edges
    sharpen_kernel = np.array([[-1,-1,-1],
                               [-1,9,-1],
                               [-1,-1,-1]]
                              )

    ddepth = -1 # Output image will have the same depth as the source.
    # Depth is the number of bits used to represent color in the image
    sharpen = cv2.filter2D(blur, ddepth, sharpen_kernel)
    if write_intermediate:
        cv2.imwrite(str(target_directory.joinpath("intermediate","sharpen_" + filename + ".jpeg")), sharpen)

    # Threshold to convert the image to a binary image.
    # threshold = 0.26*np.median(np.sort(sharpen.flatten())[::-1][0:1000])
    # threshold = 46
    threshold = 50


    # threshold = 2*np.mean(np.sort(sharpen.flatten())[0:20])
    # Lowering threashold incorporates more light colored areas
    thresh = cv2.threshold(sharpen,threshold,255, cv2.THRESH_BINARY_INV)[1]
    if write_intermediate:
        cv2.imwrite(str(target_directory.joinpath("intermediate","thresh_" + filename + ".jpeg")), thresh)

    # Perform morphological operations:
    morph_size = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size,morph_size))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=15)

    # Get the contours
    cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] # Get the contours and not the hierarchy

    image_area = gray.shape[0]*gray.shape[1] # Estimate the area of the contour
    # number_of_squares = 1
    # guessed_fraction_of_photo_occupied_by_squares = 0.25
    # expected_block_area = image_area*guessed_fraction_of_photo_occupied_by_squares/number_of_squares

    expected_block_area =831814 # Area of a block based on previous iterations

    min_area = expected_block_area*0.9
    max_area = expected_block_area*1.1

    cropped = False
    cropped_area = np.nan
    side_lengths= [np.nan]
    # Loop through all extracted contours and choose one that corresponds with the expected area of the contour
    for i,contour in enumerate(cnts):
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            print("Selected Area between min and max:",min_area, area,max_area, "Threshold: ", threshold)
            cropped_and_rotated,side_lengths = crop_rotated(contour, image,filename,fixed_sidelength = int(0.95*919)) # Do cropping of picture. Sidelength hard coded based on median side length.
            cropped = True
            cropped_area = area

            # Write the cropped image to the target directory
            cv2.imwrite(str(target_directory.joinpath("out", "cropped_and_rotated_" + filename + ".jpeg")),
                        cropped_and_rotated)
            break

    if cropped is not True:
        print("Cropping unsuccessful for:", filename,"with threshold: ",threshold)


    return cropped_area,side_lengths,threshold

r = Parallel(n_jobs=4)(delayed(do_cropping)(fname.name,source_directory,target_directory) for fname in source_directory.iterdir())
r = [list(i) for i in zip(*r)]
areas = np.array(r[0])
side_lengths = r[1]
thesholds = r[2]
side_lengths = np.array(list(itertools.chain(*side_lengths)))

print(side_lengths)
print("Median area: ", np.nanmedian(areas))
print("Median side length: ", np.nanmedian(side_lengths))
print("Median threshold: ", np.nanmedian(thesholds))

print("fraction unsuccessful:",np.sum(np.isnan(areas))/len(areas))
print("number unsuccessful:",np.sum(np.isnan(areas)))
