from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# preprocessing Generators
train_datagenerator = ImageDataGenerator(rescale = 1.0/255,
                                        shear_range = 0.2,
                                        zoom_range = 0.5,
                                        horizontal_flip = True,
                                        rotation_range=10,
                                        width_shift_range=0.2,
                                        brightness_range=[0.2,1.2],
                                        validation_split = 0.2

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
