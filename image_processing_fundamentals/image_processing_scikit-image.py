

from skimage import io 
from matplotlib import pyplot as plt 
img = io.imread('thumbnail1.jpg', as_gray=True) 
plt.imshow(img)

from skimage.transform import rescale, resize, downscale_local_mean 


# rescaling image 
rescaled_img = rescale(img,1.0/4.0, anti_aliasing=True) 

# resizing the image 
resized_img = resize(img, (200,200)) 
plt.imshow(resized_img)

plt.imshow(rescaled_img)

# downscaling the image 
downscaled_img = downscale_local_mean(img, (112, 200)) 
plt.imshow(downscaled_img)



#### filters 
from skimage.filters import roberts, sobel, scharr, prewitt 
img = io.imread('/home/thomas/Desktop/github/Image_Processing/images/wallpaper1.jpg', as_gray=True)
edge_roberts = roberts(img) 
edge_sobel = sobel(img) 
edge_scharr = scharr(img) 
edge_prewitt = prewitt(img) 

fig, axes = plt.subplots(nrows = 2, ncols = 2, sharex=True, sharey=True, figsize = (8,8)) 
ax = axes.ravel() 
ax[0].imshow(img, cmap = plt.cm.gray) 
ax[0].set_title('Original image') 

ax[1].imshow(edge_roberts, cmap = plt.cm.gray) 
ax[1].set_title('Robers edge detection') 

ax[2].imshow(edge_scharr, cmap = plt.cm.gray) 
ax[2].set_title('scharr')

ax[3].imshow(edge_sobel, cmap = plt.cm.gray) 
ax[3].set_title('sobel')

for a in ax:
    a.axis ('off') 
plt.tight_layout() 
plt.show() 

########### canny  
from skimage.feature import canny
img = io.imread('/home/thomas/Desktop/github/Image_Processing/images/wallpaper1.jpg', as_gray=True) 
edge_canny = canny(img, sigma = 3) 
plt.imshow(edge_canny)
 

###### convolution operation 


from skimage import io 
from matplotlib import pyplot as plt
img = io.imread('/home/thomas/Desktop/github/Image_Processing/images/wallpaper1.jpg', as_gray=True)

from skimgae import restoration 
import numpy as np 
psf = np.ones((3,3))/ 9  
deconvolved,_ = restoration.unsupervised_wiener(img,psf)

















