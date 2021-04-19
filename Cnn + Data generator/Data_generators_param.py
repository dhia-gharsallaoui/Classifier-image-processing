from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# preprocessing Generators
# I decided to use data generator first to improve my data because I want to limit to the provided data.
# Also, to make some processing in the data before the learning phase. 

train_datagenerator = ImageDataGenerator(rescale = 1.0/255,          # Neural network works better on normalized Data [0,1]
                                        shear_range = 0.2,           # I add this option because I thought this classifier will be used in automated navigation and due to the speed of movement the picture may be sheared. 
                                        zoom_range = 0.5,            # This will avoid dependancy on the camera position
                                        horizontal_flip = True,      #__.
                                        rotation_range=10,           #  | ==>  I add those because i assume that the camera in move 
                                        width_shift_range=0.2,       #__|
                                        brightness_range=[0.2,1.2],  # I add this to make the classifier perfom in day and night 
                                        validation_split = 0.2       # this is for splitting train data in train and validation

)

test_datagenerator = ImageDataGenerator(rescale = 1.0/255,
                                        shear_range = 0.2,
                                        zoom_range = 0.5,
                                        horizontal_flip = True,
                                        rotation_range=10,
                                        width_shift_range=0.2,
                                        brightness_range=[0.2,1.2],

)



# Load Data Parameteres
train_path='/home/dhia/data/tain'
test_path='/home/dhia/data/test'
targ_size=(120,120)
batch_gen=10


# training Parameters
input_size=[120,120,3]
batch=16
opt='adam'
n_epoch=20
callback_param = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    ]
