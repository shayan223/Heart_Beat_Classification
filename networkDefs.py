import keras


# !!!  WARNING: You probably don't have enough memory to run this   !!!
def testConvolutional(sampleCount,atrNum,numberOfClasses):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=64, 
                                  kernel_size=3, 
                                  activation='relu', 
                                  input_shape=(atrNum,1)
                                  ))
    model.add(keras.layers.Conv1D(filters=64, 
                                  kernel_size=3, 
                                  activation='relu'
                                  ))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Conv1D(filters=64,
                                  kernel_size=3, 
                                  activation='relu'
                                  ))
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(numberOfClasses, 
                                 activation='softmax'
                                 ))
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

    return model
    

# !!!  WARNING: You probably don't have enough memory to run this   !!!
def basicConvolutional(sampleCount,atrNum,numberOfClasses):
    #length of each input vector num of atributes for input layer
    model = keras.Sequential()
    model.add(keras.layers.Conv1D( 
                            filters=128,#layer output size
                            kernel_size=3,#convolutional window size (kernel size)
                            input_shape=(atrNum,1),#define input size for first layer
                            padding='valid',#no padding
                            data_format='channels_last',#defines input shape (batch, steps, channels)
                            activation='relu',#activation function
                            use_bias=True,#this layer will have a bias vector
                            bias_initializer='zeros',#init bias vector values
                            kernel_initializer='random_uniform'#init kernel values
                            ))
    #model.add(keras.layers.Flatten())#reduces data dimentionality
    model.add(keras.layers.MaxPooling1D(
                            pool_size=4,#simplify data to pool size
                            #strides=64,
                            padding='valid',
                            data_format='channels_first'
                            ))
    model.add(keras.layers.Dense(64,
                                 activation='relu'
                                 ))
    model.add(keras.layers.Conv1D( 
                            64,#layer output size
                            3,#convolutional window size (kernel size)
                            padding='valid',#no padding
                            data_format='channels_first',#defines input shape (batch, steps, channels)
                            activation='relu',#activation function
                            use_bias=True,#this layer will have a bias vector
                            bias_initializer='zeros',#init bias vector values
                            kernel_initializer='random_uniform'#init kernel values
                            ))
    model.add(keras.layers.MaxPooling1D(
                            pool_size=2,
                            padding='valid',
                            data_format='channels_first'
                            ))
    model.add(keras.layers.Flatten())#reduces data dimentionality
    model.add(keras.layers.Dense(32,
                                 activation='relu'
                                 ))
    #model.add(keras.layers.Flatten())#reduces data dimentionality
    model.add(keras.layers.Dense(numberOfClasses, activation='relu'))
    model.compile(loss='sparse_categorical_crossentropy',#define loss function
                  optimizer='adam',
                  metrics=['accuracy'])#list of metrics used to evaluate model
    
    return model


##### Currently the most memory efficient network #####
    
def Convolutional_B(sampleCount,atrNum,numberOfClasses):
    model = keras.Sequential()
    #model.add(keras.layers.MaxPooling1D(pool_size=64))
    model.add(keras.layers.Conv1D(filters=64, 
                                  kernel_size=3, 
                                  activation='relu',
                                  input_shape=(atrNum,1)
                                  ))
    model.add(keras.layers.MaxPooling1D(pool_size=64))
    model.add(keras.layers.Conv1D(filters=64, 
                                  kernel_size=3, 
                                  activation='relu'
                                  ))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPooling1D(pool_size=64))
    model.add(keras.layers.Conv1D(filters=64,
                                  kernel_size=3, 
                                  activation='relu'))
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(numberOfClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

    return model


def time_series_network(sampleCount,atrNum,numberOfClasses):
    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(units=1, 
                                     activation='tanh', 
                                     use_bias=True, 
                                     kernel_initializer='glorot_uniform', 
                                     recurrent_initializer='orthogonal', 
                                     bias_initializer='zeros', 
                                     kernel_regularizer=None, 
                                     recurrent_regularizer=None, 
                                     bias_regularizer=None, 
                                     activity_regularizer=None, 
                                     kernel_constraint=None, 
                                     recurrent_constraint=None, 
                                     bias_constraint=None, 
                                     dropout=0.0, 
                                     recurrent_dropout=0.0, 
                                     return_sequences=False, 
                                     return_state=False, 
                                     go_backwards=False, 
                                     stateful=False, 
                                     unroll=False))
    model.add(keras.layers.Dense(numberOfClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    
    return model