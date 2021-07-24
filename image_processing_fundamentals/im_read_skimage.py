

# credit - https://www.youtube.com/watch?v=52pMFnkDU-4

'''
In this i will try to load images in python by using various libraries 
PIL -- PILLOW 
matplotlib - 
skimage - 
openCV 

** for propriatery images like czi, OME-TIFF 

PILLOW 
========================
pillow is an image manipulation and processing library you can use pillow for resize, crop and to do basic filtering 
for advanced task we need computer vision libraries like openCV, scikit image and scikit learn. 

pip install Pillow 
import PIL 
''' 
########################################################
from PIL import Image 
import numpy as np # to convert images to arrays 
im_path = '/home/thomas/Desktop/github/Image_Processing/images/giti.jpg'
img = Image.open(im_path) # not a numpy array 
print(type(img))

# outptut images 
img.show()

# format of the image 
print(img.format) 

# mode of image 
print(img.mode) 
 
# note : pil is not by default numpy array but can convert pil image to numpy array 
img_array = np.asarray(img) 
print(type(img_array)) 


######################################################

# using matplotlib 

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt  
im_path = '/home/thomas/Desktop/github/Image_Processing/images/giti.jpg'
img2 = mpimg.imread(im_path)  # this is a numpy array 
print(type(img2))

print(img2)
print(img2.shape)
plt.imshow(img2)

plt.colorbar() # put color bar next to the image 

#####################################################
# using scikit image
'''
 pip install scikit-image
scikit image is an image processing library that includes algorithms, segmentation, 
geometric transformation , color space manipulation , analysis, feature extraction 
and more 
'''
  

from skimage import io, img_as_float, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt 


image = img_as_float(io.imread(im_path))

#image2 = io.imread("images/test_image.jpg").astype(np.float)
#avoid using astype as it violates assumptions about dtype range.
#for example float should range from 0 to 1 (or -1 to 1) but if you use 
#astype to convert to float, the values do not lie between 0 and 1. 
#print(image.shape)
plt.imshow(img)

print(image)

#print(image2)
#image8byte = img_as_ubyte(image)
#print(image8byte)



import cv2
path = '/home/thomas/Desktop/github/Image_Processing/images/giti.jpg'
grey_img = cv2.imread(path, 0)
color_img = cv2.imread(path, 1)

#images opened using cv2 are numpy arrays
print(type(grey_img)) 
print(type(color_img)) 

# Use the function cv2.imshow() to display an image in a window. 
# First argument is the window name which is a string. second argument is our image. 

cv2.imshow("pic", grey_img)
# cv2.imshow("color pic", color_img)

# Maintain output window until 
# user presses a key or 1000 ms (1s)
cv2.waitKey(0)          

#destroys all windows created
cv2.destroyAllWindows() 










