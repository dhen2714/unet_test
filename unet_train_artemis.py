from model_unet import unet
from keras_datagenh5 import DataGeneratorH5
import keras
from keras.optimizers import Adam


model = unet(rows=480, cols=640, channels=1)

bsize = 5
steps = 3500/bsize
valsteps = 500/bsize
num_epochs = 20

train_generator = DataGeneratorH5(
    'train_warped.h5', 
    'train_notwarped.h5', 
    batch_size=bsize,
    rescale=1/255
)
val_generator = DataGeneratorH5(
    'test_warped.h5', 
    'test_notwarped.h5', 
    batch_size=bsize,
    rescale=1/255
)

callbacks = [
keras.callbacks.TensorBoard(log_dir="log",histogram_freq=0,
                            write_graph=True,write_images=False),
keras.callbacks.ModelCheckpoint('log/model_20epochs.h5',verbose=0,save_weights_only=True)
]
adam = Adam(lr=0.00005, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=0.0, amsgrad=False)

model.compile(optimizer=adam, loss='mse', metrics=['mse'])

model.fit_generator(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=steps,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=valsteps,
    max_queue_size=100,
)