import cv2

def show_large_image(image_name,image_obj):
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)  #
    cv2.resizeWindow(image_name, 600, 600)
    cv2.imshow(image_name, image_obj)
    return
