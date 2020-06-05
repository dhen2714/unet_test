"""
Unet model.
"""
import keras


def unet(rows=None, cols=None, channels=3):
    "unet for image reconstruction"
    P0 = x = keras.layers.Input(shape=(rows, cols, channels),name="u_net_input")
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), name='conv1', use_bias=True,padding="same")(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(2, 2), name='conv1a', use_bias=True,padding="same")(x)
    P1= x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)

    x = keras.layers.Conv2D(64,(3,3),strides=(1,1),name='conv2',use_bias=True,padding="same")(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    x = keras.layers.Conv2D(64,(3,3),strides=(2,2),name='conv2a',use_bias=True,padding="same")(x)
    P2= x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    
    x = keras.layers.Conv2D(128,(3,3),strides=(1,1),name='conv3',use_bias=True,padding="same")(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    x = keras.layers.Conv2D(128,(3,3),strides=(2,2),name='conv3a',use_bias=True,padding="same")(x)
    P3= x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)

    x = keras.layers.Conv2D(128,(3,3),strides=(1,1),name='conv4',use_bias=True,padding="same")(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    x = keras.layers.Conv2D(128,(3,3),strides=(2,2),name='conv4a',use_bias=True,padding="same")(x)
    P4= x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)

    x = keras.layers.Conv2D(256,(3,3),strides=(1,1),name='conv5',use_bias=True,padding="same")(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    x = keras.layers.Conv2D(256,(3,3),strides=(2,2),name='conv5a',use_bias=True,padding="same")(x)
    P5= x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)

    x = keras.layers.Deconvolution2D(nb_filter=128, nb_row=3, nb_col=3,subsample=(1, 1),name='deconv4',border_mode='same')(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    x = keras.layers.Deconvolution2D(nb_filter=128, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv4a',border_mode='same')(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    C4= x = keras.layers.Add()([P4,x])

    x = keras.layers.Deconvolution2D(nb_filter=128, nb_row=3, nb_col=3,subsample=(1, 1),name='deconv3',border_mode='same')(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    x = keras.layers.Deconvolution2D(nb_filter=128, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv3a',border_mode='same')(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    C3= x = keras.layers.Add()([P3,x])


    x = keras.layers.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(1, 1),name='deconv2',border_mode='same')(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    x = keras.layers.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv2a',border_mode='same')(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    C2= x = keras.layers.Add()([P2,x])

    x = keras.layers.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(1, 1),name='deconv1',border_mode='same')(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    x = keras.layers.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv1a',border_mode='same')(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    C1= x = keras.layers.Add()([P1,x])

    x = keras.layers.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(1, 1),name='deconv0',border_mode='same')(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    x = keras.layers.Deconvolution2D(nb_filter=channels, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv0a',border_mode='same')(x)
    x = keras.layers.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
    C0= x = keras.layers.Add()([P0,x])

    x = keras.layers.Conv2D(channels,(3,3),strides=(1,1),name='convr',use_bias=True,padding="same")(x)
    denoised_image = keras.layers.Activation('linear')(x)

    model = keras.models.Model(inputs=P0,outputs=denoised_image)
    model.summary()
    return model 
