



################################################
from skimage import io, img_as_ubyte
img = io.imread("fundamentals/simg.jpg", as_gray=True) 
print(type(img))
print(img.dtype)
print(img.shape)

img = img_as_ubyte(io.imread("fundamentals/simg.jpg", as_gray=False)) 
print(type(img))
print(img.dtype)
print(img.shape)

##################################################
# subsetting image ( cropping ) and getting mean, max and min 
print(img[10:15, 20:25]) 
mean_grey = img.mean() 
max_value  = img.max() 
min_value = img.min() 

print("min, max, mean: ", min_value, max_value, mean_grey) 




################################################3
img = img_as_ubyte(io.imread("fundamentals/simg.jpg", as_gray=True)) 
import numpy as np 
from scipy import ndimage 

# for plotting multiple 
from matplotlib import pyplot as plt 

# flipping image left-right 
flippedLR = np.fliplr(img)

# flipping up and down 
flippedUB = np.flipud(img)

plt.subplot(2,1,1) 
plt.imshow(img, cmap = "Greys")

plt.subplot(2,2,3)
plt.imshow(flippedLR, cmap = 'Blues') 

plt.subplot(2,2,4) 
plt.imshow(flippedUB, cmap = 'hsv') 
###########################################

# rotation 
img = img_as_ubyte(io.imread("fundamentals/simg.jpg", as_gray=False)) 
rotated = ndimage.rotate(img, 45, reshape = False)
plt.imshow(rotated)
 
# matplotlib color maps 
#####################################################

# filter :
    # uniorm filter : blurring filteer 
    # gaussian filter : smoothing the noise 
        # it does not preserve edges 
    # median filter : 

uniform_filter = ndimage.uniform_filter(img, size = 9) 
plt.imshow(uniform_filter)


gaussian_filter = ndimage.gaussian_filter(img, sigma = 1) 
plt.imshow(gaussian_filter)
 

median_filter = ndimage.median_filter(img, 3) 
plt.imshow(median_filter) 
 

sobel_img = ndimage.sobel(img, axis = -1) 
plt.imshow(sobel_img)


# scipy image filters 







