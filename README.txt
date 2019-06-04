**Copy right and licensing information is in license.txt**


Project: Neural Network Classification of Heartbeats

Description: This project aims to use various neural network
implementations to create a classifier aimed to classify 
heart conditions as either extrahls, murmur, normal, or artifact
using digital stethoscope audio.

Testing: There was no direct unit testing for this project as
the focus was on the construction of a classifier, which either
fails as a whole, or correctly compiles. However different networks
may show different accuracy on the same data. Essentially testing
done in this program was through the comparison of different neural
network architectures and their performance.

Metrics: The metric for "Performance" for each network was based on
its accuracy on its validation data across all epochs of its training.

Data Processing: Currently all data is being processed as raw input, 0 padded
to the longest audio length.

Dependancies: To successfully run the program, make sure you have the
following dependancies: Keras, numpy, pandas, scipy, timeit, and matplotlib.

Running the Program: To run the program, open kerasNet.py and run the script.
the callFit variable at the top of the file can be set to 0 to simply compile
the chosen neural network and display its architecture. Setting it to 1 will 
begin training the network and display its progress and accuracy. The 
epochCount variable will determine how many epochs of training the model
will train for if callFit is set to 1.

Current progress: Currently networks testConvolutional, basicConvolutional,
and Convolutional_B all correctly compile and run, however 

future tests:




