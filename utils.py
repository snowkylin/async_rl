import cv2
from config import *

def rgb2gray(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    return res

def resize(image):
    return cv2.resize(image, (args.resize_width, args.resize_height))