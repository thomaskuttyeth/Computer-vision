# importing warning 
import warnings
warnings.filterwarnings("ignore")

# importing modules 
from preprocessing import data_loader as dl 
from model.network import Cnn_Model


# loading the cnn network and building 
weather_classifier = Cnn_Model(size = 64,channels = 3) 
weather_classifier.build() 
weather_classifier.compile()

# dataloader 
d_loader = dl.DataLoader('weather_data')
d_loader.preprocess('jpg')

# resizing the images 
d_loader.resize(d_loader.dataset,size = 64, channel = 3) 

# get the preprocessed images
df = d_loader.resized_df

# splitting the dataset into training and testing 
d_loader.splitting(ratio = 0.2, rand_state=101)

# training the model 
hist = weather_classifier.fit_model(
    X_train= d_loader.X_train, 
    y_train= d_loader.y_train, 
    batch_size = 10, 
    verbose = 1, 
    epochs = 20, 
    validation_split = 0.1,
    shuffle = False)

# getting the test accuracy 
weather_classifier.test(d_loader.X_test, d_loader.y_test)


# saving 
weather_classifier.save('weather_model')

# performance graph 
weather_classifier.visualize(hist) 


