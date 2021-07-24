#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:45:03 2021

@author: thomas
"""


# loading the library
from PIL import Image 
img_path = '/home/thomas/Desktop/github/Image_Processing/images/pexels-jack-krzysik-7753054.jpg'
img = Image.open(img_path)
img.show() 
print(img.size) 



# resizing the image and saving it in the disk 
small_img = img.resize((200,300))  

# saving the imgage in a specified directory 
small_img.save("/home/thomas/Desktop/github/Image_Processing/fundamentals/simg.jpg") 

simg = Image.open('/home/thomas/Desktop/github/Image_Processing/fundamentals/simg.jpg') 
simg.show()
simg.size

# resizing it inplace 
dir_path = '/home/thomas/Desktop/github/Image_Processing/fundamentals' 
img.thumbnail((200,300))
img.save(dir_path+'/thumbnail1.jpg') 
thumbimg = Image.open(dir_path+'/thumbnail1.jpg')
thumbimg.size  # observe the small change 
# it first take the height/width and scales accordingly keeping the aspect ratio 


##### cropping the images 
imges = '/home/thomas/Desktop/github/Image_Processing/images' 
dir =  '/home/thomas/Desktop/github/Image_Processing/fundamentals'
img = Image.open(imges+'/lake.jpg') 
img.show() 

cropped_img = img.crop((0,0, 600, 600)) 
cropped_img.save(dir+'/croppedimg.jpg')
img.size
        

##### copying 
img1 = Image.open(imges+'/lake.jpg') 
img2  =Image.open(dir+'/thumbnail1.jpg') 

img1_copy = img1.copy() 
img1_copy.paste(img2, (50,50)) 
img1_copy.save(dir +'/pasted.jpg') 

# showing the pasted image 
pasted = Image.open(dir+'/pasted.jpg') 
pasted.show() 



###### rotating the images  ( we lose the edges )
img = Image.open(imges+'/lake.jpg') 
img90 = img.rotate(90) 
img90.show() 



img = Image.open(imges+'/lake.jpg') 
img90 = img.rotate(90, expand = 'True') 
img90.show() 




##### flipping the images 
img_fliplr = img.transpose(Image.FLIP_LEFT_RIGHT) 
img_fliplr.show() 

img_fliplr = img.transpose(Image.FLIP_TOP_BOTTOM) 
img_fliplr.show() 

##### changing to grey scale 

grey_image = img.convert('L')
grey_image.show()


# https://pillow.readthedocs.io/en/stable/reference/Image.html


##### automating the task 
import glob 
from PIL import Image
import os 

# path of the image folder 
path = '/home/thomas/Desktop/github/Image_Processing/images/*'

# creating new directory for saving the rotated images 
os.mkdir('rotate_images')
os.chdir('rotate_images') 

# loop over each image in the path
for file in glob.glob(path):
    name = file.split('/')[-1]
    
    a = Image.open(file) 
    # rotating each image 
    rotated45 = a.rotate(45, expand = True) 
    rotated45.save(name+"_rotated45.png", "PNG") 
    
    
    
    
    
    
    























