import numpy as np
#from keras import layers
import keras
import scipy.io.wavfile as wavio
import timeit
import matplotlib.pyplot as plot
import seaborn as sb
import pandas as pd
import networkDefs

start = timeit.default_timer()
####Set up data reference table########
setA = pd.read_csv("set_a.csv")
setA = setA.drop(columns="sublabel")#no sublables, not necesary
setA = setA.dropna(axis=0)#remove all unlabled rows


######## Prepare training data ##########
data = []
rate = []
tempShape = []
labels = []
numOfFiles = setA.shape[0]

#read in data and labels into lists
for i in range(0,numOfFiles):#loops through all data files
#**comment the above, and uncomment the below for testing**
#for i in range(1):#arbitrary length loop for testing
    
    rateTemp, dataTemp = wavio.read(setA.loc[i,"fname"])#read from file, rate=sample rate of file
    tempShape.append(dataTemp.shape[0])
    rate.append(rateTemp)
    data.append(dataTemp)
    #normalise data between -1 and 1
    data[i] = data[i] / rate[i]
    #Extract and store lables for data
    labels.append(setA.loc[i,"label"])
    
#convert labels to numpy array for further use
labels = np.asarray(labels)
#pad all data with 0's to longest read in audio length
maxlen = max(tempShape)
for i in range(len(data)):
    padLength = maxlen - data[i].shape[0]
    data[i] = np.pad(data[i],(0,padLength),'constant')
    
    
data = np.stack(data, axis=0)

#########################################
'''
###uncomment this section for manual seperation of ###
###data set to training and testing data ###########

###### Seperate testing and training data ######

#shuffle list of data then use the random 
#numbers to map training and test data
#to the original data array
length = data.shape[0]
halfWay = int((length)/2)
shuffle = np.arange(data.shape[0])
np.random.shuffle(shuffle)
mapTrain = shuffle[0:halfWay]#mapping to shuffled training data
mapTest = shuffle[halfWay:length]#mapping to shuffled testing data
training = np.zeros((halfWay,data.shape[1]))
testing = np.zeros((halfWay,data.shape[1]))
trainingLabels = []#np.zeros(halfWay)
testingLabels = []

for i in range(halfWay-1):
    training[i] = data[mapTrain[i]]
    testing[i] = data[mapTest[i]]
    trainingLabels.append(labels[mapTrain[i]])
    testingLabels.append(labels[mapTest[i]])
    if(mapTrain[i] == mapTest[i]):
        print("ERROR: This should not happen")
    

#########################################
'''
##### Convert Label list to numbers #####
#0 = "artifact"
#1 = "extrahls"
#2 = "murmur"
#3 = "normal"
labelsAsNum = []
classCount = 4
for i in labels:
    if(i == 'artifact'):
        labelsAsNum.append(0)
    if(i == 'extrahls'):
        labelsAsNum.append(1)
    if(i == 'murmur'):
        labelsAsNum.append(2)
    if(i == 'normal'):
        labelsAsNum.append(3)

labelsAsNum = np.asarray(labelsAsNum)#convert to numpy array

###### Free unnecesary data before training network ######

del(setA)
del(labels)
del(tempShape)

###### Define neural network ############

atrNum = maxlen#number of atributes per entry
dataCount = data.shape[0]#number of data entries
networkA = networkDefs.Convolutional_B(dataCount,atrNum,classCount)

###### Train neural network ############
#Reshape data to 3d for network
#Network expects (BatchSize,NumberofSamples,dataitem)
data = np.expand_dims(data,axis=2)
labelsAsNum = np.expand_dims(labelsAsNum,axis=1)


networkA.summary()
#print(labelsAsNum.shape, data.shape)

resultsA = networkA.fit(
        data,#data list
        labelsAsNum,#label list
        batch_size=None,#batch size
        epochs=10,#number of training epochs
        verbose=1,#verbosity of training progress
        callbacks=None,#call back function
        validation_split=.3,#how much data is testing vs training
        validation_data=None,#if testing and training data is seperate
        shuffle=True#whether or not to shuffle data between epochs
        #initial_epoch=None,#epoch to continue from if resuming past training
        #steps_per_epoch=None,#number of samples trained per epoch (None = all)
        #validation_steps=None,#number of batches to validate before stopping
        #validation_freq=1#how often to validate, takes int or list of points
        )

########################################
end = timeit.default_timer()
print("Run time (seconds): ",end-start)#calculates total run time
'''

########################################
end = timeit.default_timer()
print("Run time (seconds): ",end-start)#calculates total run time
plot.plot(graphX,graphY)#plots graph
'''


'''
######## Prepare data set B ##########
setB = pd.read_csv("set_b_eddited.csv")
setB = setB.drop(columns="sublabel")#no sublables, not necesary
setB = setB.dropna(axis=0)#remove all unlabled rows
print(setB)

dataB = []
rateB = []
tempShapeB = []
labelsB = []
numOfFilesB = setB.shape[0]

#read in data and labels into lists
print(setB)
print(setB.shape)
#print(setB[400])
for i in range(0,numOfFilesB):#loops through all data files
#**comment the above, and uncomment the below for testing**
#for i in range(1):#arbitrary length loop for testing
    
    #print(i,numOfFilesB)
    #print(setB.loc[i,"fname"])
    
    rateTemp, dataTemp = wavio.read(setB.loc[i,"fname"])#read from file, rate=sample rate of file
    tempShape.append(dataTemp.shape[0])
    rateB.append(rateTemp)
    dataB.append(dataTemp)
    #normalise data between -1 and 1
    dataB[i] = dataB[i] / rateB[i]
    #Extract and store lables for data
    labelsB.append(setB.loc[i,"label"])
    

#pad all data with 0's to longest read in audio length
maxlenB = max(tempShapeB)
for i in range(len(dataB)):
    padLength = maxlenB - dataB[i].shape[0]
    dataB[i] = np.pad(dataB[i],(0,padLength),'constant')
    
print(dataB[5].shape)
    
dataB = np.stack(dataB, axis=0)
print(dataB.shape)
print(labelsB)
#########################################
'''





