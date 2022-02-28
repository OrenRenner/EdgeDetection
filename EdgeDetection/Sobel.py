import cv2
import numpy as np

'''

SobelEdges -> return image with edges

    image - input image
    flag - type of return edge image, default -1 
        -1 -> return all edge images SobelX, SobelY, SobelXY
        0 -> return SobelX
        1 -> return SobelY
        2 - return SobelXY
        
    blur - type of blur, default None, without bluring
        'gaus' - try to use GaussianBlur
        'median' - try to use MedianBlur
        
    kernel - size of kernel of blur, default (3, 3)

'''


def SobelEdges(image: np.ndarray, flag=-1, blur=None, kernel=(3,3)):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur is None:
        img_blur = img_gray
    elif blur == 'gaus':
        img_blur = cv2.GaussianBlur(img_gray, kernel, 0)
    elif blur == 'median':
        img_blur = cv2.medianBlur(img_gray, kernel)
    else:
        raise ValueError("Undefined blur!")

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    if flag == -1:
        return sobelx, sobely, sobelxy
    elif flag == 0:
        return sobelx
    elif flag == 1:
        return sobely
    else:
        return sobelxy