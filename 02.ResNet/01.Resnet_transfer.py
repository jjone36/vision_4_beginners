import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from keras.layers import Flatten, Dense
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
#from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

# The relevant dataset can be found here: https://www.kaggle.com/paultimothymooney/blood-cells
# directory path for train and test set
train_dir = '../images/blood_cell_images/TRAIN'
valid_dir = '../images/blood_cell_images/TEST'

# Load the images
folders = glob(train_dir + '/*')
tr = glob(train_dir + '/*/*.jp*g')   # / "fruit_name" / "fruit image file"
val = glob(valid_dir + '/*/*.jp*g')

# Modeling
n_class = len(fruits)
im_size = 224

epochs = 5
batch_size = 32

# Loading a pre-trained model
res = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
#vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = (im_size, im_size, 3))

# Do not train the weights
for layer in res.layers:
    layer.trainable = False

# Build additional layers
x = res.output
x = Flatten()(x)
output = Dense(n_class, activation = 'softmax')(x)

# Create a model
model = Model(inputs = res.input, outputs = output)
model.summary()

# Compile the model
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metircs = ['accuracy'])


# Image Augmentation
gen = ImageDataGenerator(rotation_range = 20,
                        shear_range = .1,
                        zoom_range= .2,
                        width_shift_range = .1, height_shift_range = .1,
                        horizontal_flip=True, vertical_flip=True,
                        preprocessing_function= preprocess_input)

train_generator = gen.flow_from_directory(train_dir,
                                          target_size=(im_size, im_size),
                                          batch_size = batch_size)

valid_generator = gen.flow_from_directory(valid_dir,
                                          target_size= (im_size, im_size),
                                          batch_size=batch_size)

# Fit the model
r = model.fit_generator(generator = train_generator,
                        epochs = epochs,
                        steps_per_epoch = len(tr) // batch_size,
                        validation_data= valid_generator,
                        validation_steps= len(val) // batch_size)

# Evaluation
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()


# Plot confusion matrix
def get_confusion_matrix(data_path, N):
    predictions = []
    targets = []
    i = 0
    generated_data = gen.flow_from_directory(data_path,
                                            target_size = (im_size, im_size),
                                            batch_size = batch_size *2)
    for x, y in generated_data:
        # the probabilies against each class (the outcome of softmax)
        pred = model.predict(x)
        # OneHotEncoding
        pred = np.argmax(pred, axis = 1)
        y = np.argmax(y, axis = 1)

        predictions = np.concatenate((predictions, pred))
        targets = np.concatenate((targets, y))

        # Breaking infinite loop
        if len(targets) >= N:
            break

        # Process tracking
        i += 1
        if i % 50 == 0:
            print("========Processing: %.2f" % (i/N))

    cm = confusion_matrix(targets, predictions)
    return cm

# Get the confusion matrix
tr_cm = get_confusion_matrix(train_dir, len(tr))
valid_cm = get_confusion_matrix(valid_dir, len(val))
print("========The confusion matrix for the train set: \n")
print(tr_cm)
print("========The confusion matrix for the validation set: \n")
print(valid_cm)
