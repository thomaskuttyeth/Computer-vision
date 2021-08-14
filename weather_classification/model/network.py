


from tensorflow import keras 
import matplotlib.pyplot as plt 
import numpy as np 
class Cnn_Model():
    def __init__(self,size,channels):
        self.size = size
        self.channels = channels
        self.input_ = None 
        self.model = None 
        self.input_shape = (self.size,self.size,self.channels)
    def build(self):       
        self.input_ = keras.layers.Input(self.input_shape) 

        # first block 
        conv1 = keras.layers.Conv2D(32,(3,3), activation= 'relu', padding = 'same') (self.input_) 
        pool1 = keras.layers.MaxPooling2D(pool_size = (2,2))(conv1) 
        norm1 = keras.layers.BatchNormalization(axis = -1)(pool1) 
        drop1 = keras.layers.Dropout(rate = 0.2)(norm1) # adding regularization 
        
        # second block 
        conv2 = keras.layers.Conv2D(32,(3,3), activation= 'relu', padding = 'same')(drop1) 
        pool2 = keras.layers.MaxPooling2D(pool_size = (2,2))(conv2) 
        norm2 = keras.layers.BatchNormalization(axis = -1)(pool2) 
        drop2 = keras.layers.Dropout(rate = 0.2)(norm2) # adding regularization 

        # flattening weights 
        flatten = keras.layers.Flatten()(drop2) 

        # fully connected set 1  
        hidden1 = keras.layers.Dense(512, activation = 'relu')(flatten)
        norm3 = keras.layers.BatchNormalization(axis = -1)(hidden1)
        drop3 = keras.layers.Dropout(rate = 0.2)(norm3) 


        # fully connected set2
        hidden2 = keras.layers.Dense(512, activation = 'relu')(drop3)
        norm4 = keras.layers.BatchNormalization(axis = -1)(hidden2)
        drop4 = keras.layers.Dropout(rate = 0.2)(norm4)

        # output layer 
        self.output_ = keras.layers.Dense(4,activation = 'softmax')(drop4) 

    def compile(self):
        # model compiling 
        self.model = keras.Model(inputs = self.input_, outputs =self.output_)
        self.model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy']) 
    
    def fit_model(self, X_train,y_train,batch_size,verbose,epochs,validation_split, shuffle):
        history = self.model.fit(np.array(X_train),
                 y_train, 
                 batch_size=batch_size,
                 verbose = verbose, 
                 epochs = epochs,
                 validation_split = validation_split, 
                 shuffle =shuffle)
        return history
    
    def get_summary(self):
        return self.model.summary() 
    
    # getting testing result 
    def test(self,X_test,y_test):
        print('Test_Accuracy: {:.2f}%'.format(self.model.evaluate(np.array(X_test), np.array(y_test))[1]*100))
    
    def save(self,name_model):
        mdl ="{}.h5".format(name_model) 
        self.model.save(mdl)
    
    # model performance visualization 
    def visualize(self,history):
        f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4 ))
        f.suptitle('CNN PERFORMANCE', fontsize = 12) 
        f.subplots_adjust(top = 0.85, wspace = 0.3) 
        
        max_epoch = len(history.history['accuracy'])+1 
        epoch_list = list(range(1, max_epoch)) 
        ax1.plot(epoch_list, history.history['accuracy'], label = 'Train Accuracy') 
        ax1.plot(epoch_list, history.history['val_accuracy'], label = 'Validation Accuracy') 
        ax1.set_xticks(np.arange(1, max_epoch, 5)) 
        ax1.set_ylabel('ACCCURACY VALUE') 
        ax1.set_xlabel('EPOCH') 
        ax1.set_title('ACCURACY') 
        ax1.legend(loc = 'best') 
        
        
        ax2.plot(epoch_list, history.history['loss'], label = 'Train loss') 
        ax2.plot(epoch_list, history.history['val_loss'], label = 'Validation loss') 
        ax2.set_xticks(np.arange(1, max_epoch, 5)) 
        ax2.set_ylabel('LOSS VALUE') 
        ax2.set_xlabel('EPOCH') 
        ax2.set_title('LOSS') 
        ax1.legend(loc = 'best')        









