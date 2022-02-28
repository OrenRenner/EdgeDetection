import cv2
import scipy as sp
import numpy as np
import scipy.ndimage as nd
from scipy import signal

'''

Laplacian Operator

    From the explanation above, we deduce that the second derivative can be used to detect edges. Since images are "*2D*", we would need to take the derivative in both dimensions. Here, the Laplacian operator comes handy.

    The Laplacian operator is defined by:

    Laplace(f)=∂2f∂x2+∂2f∂y2
    The Laplacian operator is implemented in OpenCV by the function Laplacian() . In fact, since the Laplacian uses the gradient of images, it calls internally the Sobel operator to perform its computation.

The arguments are:

    src_gray: The input image.
    dst: Destination (output) image
    ddepth: Depth of the destination image. Since our input is CV_8U we define ddepth = CV_16S to avoid overflow
    kernel_size: The kernel size of the Sobel operator to be applied internally. We use 3 in this example.
    scale, delta and BORDER_DEFAULT: We leave them as default values.

'''


def LaplacianEdges(image: np.ndarray, ddepth=cv2.CV_16S, kernel_size=3, blur=None, kernel=(3,3)):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur is None:
        img_blur = img_gray
    elif blur == 'gaus':
        img_blur = cv2.GaussianBlur(img_gray, kernel, 0)
    elif blur == 'median':
        img_blur = cv2.medianBlur(img_gray, kernel)
    else:
        raise ValueError("Undefined blur!")

    # Apply Laplace function
    dst = cv2.Laplacian(img_blur, ddepth, ksize=kernel_size)

    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)

    return abs_dst


def LoG(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    LoG = nd.gaussian_laplace(gray, 2)
    thres = np.absolute(LoG).mean() * 0.75
    output = sp.zeros(LoG.shape)
    w = output.shape[1]
    h = output.shape[0]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = LoG[y - 1:y + 2, x - 1:x + 2]
            p = LoG[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thres) and zeroCross:
                output[y, x] = 1

    return output


def DoG(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    A = np.array([[0, 0, -1, -1, -1, 0, 0],
                  [0, -2, -3, -3, -3, -2, 0],
                  [-1, -3, 5, 5, 5, -3, -1],
                  [-1, -3, 5, 16, 5, -3, -1],
                  [-1, -3, 5, 5, 5, -3, -1],
                  [0, -2, -3, -3, -3, -2, 0],
                  [0, 0, -1, -1, -1, 0, 0]],
                 dtype=np.float32)
    ratio = gray.shape[0] / 500.0
    new_width = int(gray.shape[1] / ratio)
    nimg = cv2.resize(gray, (new_width, 500))

    I1 = sp.signal.convolve2d(nimg, A)
    I1 = np.absolute(I1)
    I1 = (I1 - np.min(I1)) / float(np.max(I1) - np.min(I1))

    return I1
