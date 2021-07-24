



from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from simple_preprocessor import SimplePreprocessor 
from dataset_loader import SimpleDatasetLoader


from imutils import paths 
import argparse 


# construct argument parse and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,help = "path to input dataset") 

ap.add_argument("-k", "--neighbours", type = int, default = 1, help = "# of neighbours") 

ap.add_argument("-j", "--jobs", type = int, default = -1, help = "# of jobs for knn distance(-1 uses all availbale cores)") 
args = vars(ap.parse_args()) 


# grab the list of images that we'll be describing 
print("[INFO] loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))  

# intialize the image preprocessor , load the dataset from the disk 
# and reshape the data matrix 
sp = SimplePreprocessor(32, 32) 
sd1 = SimpleDatasetLoader(preprocessors=[sp]) 
(data, labels) = sd1.load(imagePaths, verbose=500) 

# show some information on memory consumptions of the images 
print("[INFO] features : {:.1f}MB".format(data.nbytes/(1024*1000.0))) 




# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)




# train and evalutate a knn classifier on the row pixel intensities 
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),target_names=le.classes_))
