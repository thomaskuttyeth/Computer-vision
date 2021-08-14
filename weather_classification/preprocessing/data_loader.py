
# importing the libraries 
import numpy as np 
np.random.seed(100) 
import os 
from PIL import Image 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical 

class DataLoader:
    def __init__(self, img_directory):
        self.dir = img_directory
        self.dataset = []
        self.labels = []
        self.resized_df = []
        self.X_train = True 
        self.X_test = True
        self.y_train = True 
        self.y_test = True 

    def get_categories(self):
        # reading the image 
        img_categories = [] 
        for i in os.listdir(self.dir):
            img_categories.append(i)
        return img_categories
    
    def preprocess(self, img_format):                                 
        for c in range(len(self.get_categories())):   
            # getting the image classes from the main data directory 
            img_class = os.listdir(self.dir+'\\'+self.get_categories()[c]) 
        
            for i, image_name in enumerate(img_class):
                if (image_name.split('.'))[1] == img_format:
                    
                    # opening images using pillow 
                    image = Image.open(self.dir+"\\"+self.get_categories()[c]+'\\'+image_name)          
                    # appending the resized image and the corresponding labels 
                    self.dataset.append(np.array(image))
                    self.labels.append(c)          
                    
    # resizing the images in the directory 
    def resize(self,dataframe, size, channel):
        for img in dataframe:
            img.resize((size,size,channel), refcheck=False) 
            self.resized_df.append(img)

        
    # splitting the dataset 
    def splitting(self, ratio, rand_state):
        self.X_train, self.X_test, self.y_train, self.y_test =train_test_split(
            self.resized_df, 
            to_categorical(np.array(self.labels)),
            test_size = ratio, 
            random_state = rand_state)
        self.X_train = self.X_train/255.0 
        self.y_train = self.y_train/255.0 
