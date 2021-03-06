import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt 

path = input("Enter the path of the image\n")
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#reducing size of the image
#img = cv2.resize(img, None, fx=0.5, fy=0.5)         #tesseract works better if the text is smaller


#Gaussian Blur

averaging_kernel = np.ones((3,3),np.float32)/9 
filtered_image = cv2.filter2D(img,-1,averaging_kernel) 
#get a one dimensional Gaussian Kernel 
gaussian_kernel_x = cv2.getGaussianKernel(5,1) 
gaussian_kernel_y = cv2.getGaussianKernel(5,1) 
#converting to two dimensional kernel using matrix multiplication 
gaussian_kernel = gaussian_kernel_x * gaussian_kernel_y.T 
#you can also use cv2.GaussianBLurring(image,(shape of kernel),standard deviation) instead of cv2.filter2D 
f_img = cv2.filter2D(img,-1,gaussian_kernel) 


#adaptive threshold

adaptive_threshold = cv2.adaptiveThreshold(f_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 11)

coords = np.column_stack(np.where(adaptive_threshold < 255))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle
(h, w) = adaptive_threshold.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(adaptive_threshold, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

text = pytesseract.image_to_string(rotated)

print(text)
cv2.imshow("Result", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
