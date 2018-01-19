from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers import Conv2D, AveragePooling2D,Activation
from keras.layers.normalization import BatchNormalization

# basic net
def getModel(model="nVidiaModel"):
    if model == "nVidiaModel":
        return nVidiaModel()
    elif model == "commaAiModel":
        return commaAiModel()
    elif model == "commaAiModelPrime":
        return commaAiModelPrime()
    elif model == "MobileNet":
        return MobileNet()
def nVidiaModel():
    """
    Creates nVidia autonomous car  model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Conv2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Conv2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def commaAiModel(time_len=1):
    """
    Creates comma.ai autonomous car  model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def commaAiModelPrime(time_len=1):
    """
    Creates comma.ai enhanced autonomous car  model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    # Add three 5x5 convolution layers (output depth 64, and 64)
    model.add(Conv2D(16, 8, 8, subsample=(4, 4), border_mode="same", W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same", W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same", W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Conv2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Flatten())

    # model.add(Dropout(.2))
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())

    # model.add(Dropout(0.50))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())

    # model.add(Dropout(0.50))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Dense(1))

    # model.compile(optimizer="adam", loss="mse")
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    return model    

def MobileNet():
    """
    Creates Google Mobile Net Models model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    #Conv / s2 3 × 3 × 3 × 32 224 × 224 × 3model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Conv2D(32, (3, 3),subsample=(2,2),border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv dw / s1 3 × 3 × 32 dw 112 × 112 × 32
    model.add(Conv2D(32, 3, 3,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv / s1 1 × 1 × 32 × 64 112 × 112 × 32
    model.add(Conv2D(64, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv dw / s2 3 × 3 × 64 dw 112 × 112 × 64
    model.add(Conv2D(64, 3, 3,subsample=(2,2),border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #Conv / s1 1 × 1 × 64 × 128 56 × 56 × 64
    model.add(Conv2D(128, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #Conv dw / s1 3 × 3 × 128 dw 56 × 56 × 128
    model.add(Conv2D(128, 3, 3,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv / s1 1 × 1 × 128 × 128 56 × 56 × 128
    model.add(Conv2D(128, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv dw / s2 3 × 3 × 128 dw 56 × 56 × 128
    model.add(Conv2D(128, 3, 3,subsample=(2,2),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv / s1 1 × 1 × 128 × 256 28 × 28 × 128
    model.add(Conv2D(256, 1, 1,subsample=(1,1),border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv dw / s1 3 × 3 × 256 dw 28 × 28 × 256
    model.add(Conv2D(256, 3, 3,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv / s1 1 × 1 × 256 × 256 28 × 28 × 256
    model.add(Conv2D(256, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv dw / s2 3 × 3 × 256 dw 28 × 28 × 256
    model.add(Conv2D(256, 3, 3,subsample=(2,2),border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv / s1 1 × 1 × 256 × 512 14 × 14 × 256
    model.add(Conv2D(512, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #5× Conv dw / s1 3 × 3 × 512 dw 14 × 14 × 512
    #Conv / s1 1 × 1 × 512 × 512 14 × 14 × 512

    model.add(Conv2D(512, 3, 3,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, 3, 3,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, 3, 3,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, 3, 3,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, 3, 3,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv dw / s2 3 × 3 × 512 dw 14 × 14 × 512
    model.add(Conv2D(512, 3, 3,subsample=(2,2),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv / s1 1 × 1 × 512 × 1024 7 × 7 × 512
    model.add(Conv2D(1024, 3, 3,subsample=(1,1),border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv dw / s2 3 × 3 × 1024 dw 7 × 7 × 1024
    model.add(Conv2D(1024, 3, 3,subsample=(2,2),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv / s1 1 × 1 × 1024 × 1024 7 × 7 × 1024
    model.add(Conv2D(1024, 1, 1,subsample=(1,1),border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Avg Pool / s1 Pool 7 × 7 7 × 7 × 1024
    model.add(AveragePooling2D(pool_size = (7, 7),strides = (2, 2),padding = 'same'))
    model.add(Flatten())
    model.add(Dropout(0.5))

    #FC / s1 1024 × 1000 1 × 1 × 1024
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model
