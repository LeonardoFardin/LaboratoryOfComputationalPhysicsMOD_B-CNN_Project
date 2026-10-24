import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout


def build_model(L, n_class=6, NF=5, kernel_size=11, l2_reg=0.2, stddev=0.05):

    input_shape = (L, 2)
    
    reg = tf.keras.regularizers.l2(l2_reg)
    ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev, seed=None)
    
    model = Sequential([
        Conv1D(filters=NF, kernel_size=kernel_size,
               kernel_initializer=ini,
               kernel_regularizer=reg,
               padding='same',
               activation='relu',
               input_shape=input_shape),
        MaxPooling1D(pool_size=2), #we need to catch a signal that is above the noise
        Conv1D(filters=NF * 2, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(n_class, activation='softmax')
    ])
    
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    
    return model