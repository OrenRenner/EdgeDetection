import cv2
import numpy as np

'''

CannyEdges -> return image with edges

    image - input image
    threshold1 and threshold2 -> thresholds for edges

    blur - type of blur, default None, without bluring
        'gaus' - try to use GaussianBlur
        'median' - try to use MedianBlur

    kernel - size of kernel of blur, default (3, 3)

'''


def CannyEdges(image: np.ndarray,  threshold1: int, threshold2: int, blur=None, kernel=(3, 3)):
    if not isinstance(threshold1, int) or not isinstance(threshold2, int):
        raise TypeError("Set threshold int value")

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur is None:
        img_blur = img_gray
    elif blur == 'gaus':
        img_blur = cv2.GaussianBlur(img_gray, kernel, 0)
    elif blur == 'median':
        img_blur = cv2.medianBlur(img_gray, kernel)
    else:
        raise ValueError("Undefined blur!")

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2)
    return edges