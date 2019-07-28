# Artificial Neural Network

# Installing Keras, Tensorflow and Theano
# conda install -c conda-forge keras

# Given that theere are only images will not have a data prepocessing part

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # To inicialize the Neural Network as a sequence of layers
from keras.layers import Conv2D # To use in the first step; the Convolution Step. 2D because of images
from keras.layers import MaxPooling2D # Step 2: Pooling
from keras.layers import Flatten # Step 3: Flattening
from keras.layers import Dense # Add the fully connected layers and a classic ANN

# Adding a timer
from timeit import default_timer as timer
start = timer()

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution layer  
# Convolution2D: We create 32 feature detectors of 3x3 dimensions. We will obtain 32 feature maps
# input_shape: format of the images.Size 64x64 and 3 because they are coloured ones  
# activation = 'relu' to get nonlinearity
classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

# Step 2 - Pooling
# Applying Max Pooling to reduce the size of future maps and therefore reduce the number
# of nodes in future fully connected layers and improving performance
# pool_size = (2,2). With 2x2 we keep the information and we are still being precise
# on where we have the high numbers in the feature maps.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer to improve the accuracy of the model. 
# We do not need the input_shape param because we already have the inputs from the 
# layer before.
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# Putting all the numbers from the feature maps cells into one single vector, we still
# keep the feature maps high numbers in this vector which represent the spacial structure
# of the input image 
# Step 3 - Flattening
classifier.add(Flatten()) 

# Step 4 - Full connection
# Making a classic ANN composed of some fully connected layers. We will use the 
# input vector as the input layer of a classic ANN which is a great classifier for
# non linear problems and image classification is a nonlinear problem. We will create
# a hidden layer which is the fully connected layer. And then the output layer 
# composed just by 1 node because this is a binary outcome (cat or dog).
# param output_dim = 128 number of nodes in the hidden layer. It should be a number between 
# the input nodes and the output ones but here there are too much inputs. so it must be 
# a number that is not too small to make the classifier a good model and not too 
# big to not make it higly compute intensive. Around 100 goes well for this model
# but it is better if it is a power of 2 so thats why the 128.
# param activation = 'relu' as this is a hidden layer.
classifier.add(Dense(units = 128, activation = 'relu')) # Hidden layer
# param activation = 'sigmoid' to return the probabilities of each class. Sigmoid
# because we have a binary outcome. for a >2 outcome we will use the Soft Max
# param output_dim = 1 Just 1 node, the predicted probability of one class. 
classifier.add(Dense(units = 1, activation = 'sigmoid')) # Output layer

# Compiling the CNN
# Compiling the whole thing by choosing stochastic gradiend descent algorithm, a
# loss function and a performance metric.
# param optimizer = 'adam' stochastic gradient descent algorithm.
# param loss = 'binary_crossentropy' loss function. If we had >2 outcomes 'categorical_crossentropy'
# param metrics = ['accuracy'] to choose the performance metric
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

# We will preprocess the images to prevent overfitting. If we dont, we will get a great 
# accuracy with the training set but a lower accuracy on the test set.
# We will use a ready to use code from the Keras documentation. https://keras.io/preprocessing/image/
# We will use an Image Augmentation trick which will create many batches of our images 
# and then to each bach will be applyed transformations like rotating or shearing. This
# way we will enrich our trainset without adding more images. 
# We will use the flow_from_directory because we have the data structured that way
from keras.preprocessing.image import ImageDataGenerator

# Transformations to apply
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# target_size must be the same as the one we specified before.
# class_mode = 'binary' because we have a binary outcome.
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

# Elapsed time in minutes
end = timer()
print("Elapsed time in minutes")
print(0.1*round((end - start)/6))

# End of work message
import os
os.system('say "your program has finished"')














