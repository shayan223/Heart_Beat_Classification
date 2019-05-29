import keras

def testConvolutional(sampleCount,atrNum,numberOfClasses):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(atrNum,1)))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(numberOfClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
'''    
def basicConvolutional(sampleCount,atrNum,numberOfClasses):
    #length of each input vector num of atributes for input layer
    model = keras.Sequential()
    model.add(keras.layers.Conv1D( 
                            4098,#layer output size
                            128,#convolutional window size (kernel size)
                            input_shape=(sampleCount,atrNum),#define input size for first layer
                            padding='valid',#no padding
                            data_format='channels_first',#defines input shape (batch, steps, channels)
                            activation='relu',#activation function
                            use_bias=True,#this layer will have a bias vector
                            bias_initializer='zeros',#init bias vector values
                            kernel_initializer='random_uniform'#init kernel values
                            ))
    #model.add(keras.layers.Flatten())#reduces data dimentionality
    model.add(keras.layers.Dense(
                            2048,#fully connected first layer 
                            #input_shape=(sampleCount,atrNum),#define input size for first layer
                            activation='relu'
                            ))
    model.add(keras.layers.MaxPooling1D(
                            pool_size=1024,#simplify data to pool size
                            strides=64,
                            padding='valid',
                            data_format='channels_first'
                            ))
    model.add(keras.layers.Dense(1024,
                                 activation='relu'
                                 ))
    model.add(keras.layers.Conv1D( 
                            1024,#layer output size
                            128,#convolutional window size (kernel size)
                            padding='valid',#no padding
                            data_format='channels_first',#defines input shape (batch, steps, channels)
                            activation='relu',#activation function
                            use_bias=True,#this layer will have a bias vector
                            bias_initializer='zeros',#init bias vector values
                            kernel_initializer='random_uniform'#init kernel values
                            ))
    model.add(keras.layers.MaxPooling1D(
                            pool_size=128,
                            padding='valid',
                            data_format='channels_first'
                            ))
    #model.add(keras.layers.Flatten())#reduces data dimentionality
    model.add(keras.layers.Dense(64,
                                 activation='relu'
                                 ))
    #model.add(keras.layers.Flatten())#reduces data dimentionality
    model.add(keras.layers.Dense(numberOfClasses, activation='relu'))
    model.compile(loss='categorical_crossentropy',#define loss function
                  optimizer='adam',
                  metrics=['accuracy'])#list of metrics used to evaluate model
    
    return model

'''