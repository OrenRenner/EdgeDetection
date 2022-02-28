import cv2
import EdgeDetection

img = cv2.imread("images/test.jpg")

sobelx, sobely, sobelxy = EdgeDetection.SobelEdges(img, blur='gaus')

# Display Sobel Edge Detection Images
cv2.imwrite('results/SobelX.jpg', sobelx)
cv2.imwrite('results/SobelY.jpg', sobely)
cv2.imwrite('results/SobelXY.jpg', sobelxy)

canny = EdgeDetection.CannyEdges(img, 100, 200, blur='gaus')
# Display Canny Edge Detection Image
cv2.imwrite('results/Canny.jpg', canny)

laplasian = EdgeDetection.LaplacianEdges(img, blur='gaus')
# Display Canny Edge Detection Image
cv2.imwrite('results/Laplacian.jpg', laplasian)

log = EdgeDetection.LoG(img)
# Display Canny Edge Detection Image
cv2.imwrite('results/LoG.jpg', log)

dog = EdgeDetection.DoG(img)
# Display Canny Edge Detection Image
cv2.imwrite('results/DoG.jpg', dog)



cv2.waitKey(0)