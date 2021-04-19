# Importing Required Libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import numpy as np
import random
from Data_generators_param import *
from skimage.io import imread, imshow, imread_collection, concatenate_images


train_data = train_datagenerator.flow_from_directory(train_path,
                                                    target_size = targ_size,
                                                    batch_size = batch_gen,
                                                    class_mode = 'categorical',
                                                    subset='training'
                                                     )

val_data = train_datagenerator.flow_from_directory(train_path,
                                                    target_size = targ_size,
                                                    batch_size = batch_gen,
                                                    class_mode = 'categorical',
                                                    subset='validation'
                                                    )

test_data = test_datagenerator.flow_from_directory(test_path,
                                                    target_size = targ_size,
                                                    batch_size = batch_gen,
                                                    class_mode = 'categorical',
                                                    )


# Plot some Geneerated Images:

# 3. visualize
fig, ax = plt.subplots(2, 3, figsize=(10, 7))
ax = ax.ravel()
plt.tight_layout()
fig.suptitle('This is some augmented pictures', fontsize=16)

for i in range(3):

    ax[i].imshow(train_data.next()[0][i])

    ax[i + 3].imshow(train_data.next()[0][i])

plt.show()



# CNN Model
cnn = tf.keras.models.Sequential()
# Convolution
cnn.add(tf.keras.layers.Conv2D(filters=64,padding = "same",kernel_size=3,activation='relu',input_shape=input_size))
cnn.add(tf.keras.layers.Conv2D(filters=32,padding = "same",kernel_size=3,activation='relu'))
# pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# Convolution
cnn.add(tf.keras.layers.Conv2D(filters=16,padding = "same",kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=16,padding = "same",kernel_size=3,activation='relu'))
#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#flaterning
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
# Output layer
cnn.add(tf.keras.layers.Dense(units=2,activation='softmax'))
# end Model
# Compiling the CNN
cnn.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
history = cnn.fit(x = train_data, validation_data = val_data, epochs = n_epoch,
                  callbacks=callback_param)

# Plot:
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

results0 = cnn.evaluate(train_data, verbose=0)
print("\n Results on training Set:")
print("     Train Loss: {:.5f}".format(results0[0]))
print("     Train Accuracy: {:.2f}%".format(results0[1] * 100))

results = cnn.evaluate(test_data, verbose=0)
print("\n Results on test Set:")
print("     Test Loss: {:.5f}".format(results[0]))
print("     Test Accuracy: {:.2f}%".format(results[1] * 100))
