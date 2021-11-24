# DATA SET

Firstly, we searched online how to manage the creation of two different generators (one for the training set and the other for the validation set) from the unique folder provided on Kaggle (“MaskDataset/training”).
There are 2 different solutions for this kind of problem:
1.	The first one with a method called `flow_from_directory()`. It implies that the folders are structured in such a way the photos we want to use for training are in a directory divided in 3 subdirectories named as the 3 output classes (0, 1, 2). The same for the validation set. 
2.	The second method, the one we opted for, assumes the use of a function called `flow_from_dataframe()` and thus also the use of Pandas’ DataFrames. We created only one “ImageDataGenerator” with the option “validation_split=0.2”, in such a way from that generator we could then instantiate training and validation generators.

# CUSTOM MODELS

After this preliminary step, we organized our code to reproduce the simple custom convolutional neural network seen during the lectures. After training, it reached an accuracy of about 70% on classification.
We did different trials with the custom convolutional neural networks, but we were not able to overcome 75% accuracy.
We also took a look to the structure of VGG16 in order to get some ideas on how to improve the custom CNN. At first hand, we did some trials but without the results we expected. 

# TRANSFER LEARNING

We finally switched to transfer learning with fine tuning, and we focused on the different choices of hyperparameters and training parameters. In particular, we tried different configurations of the followings:
1.	Learning rate
2.	Batch size
3.	Structure of the classification layer (the FC)
4.	Fine tuning or pure transfer learning and how many layers to be freezed
5.	Regularization solution and more precisely dropout and early stopping

We chose different models on which performing transfer learning:
* VGG16
* MobileNet V2

## VGG16

The first network on which we performed transfer learning was VGG16. We ran different models and we observed that:
* The network performs generally better with a learning rate of 10^-4. Practically, all of the trials we made with it used to have higher accuracy and used to reach their best performance faster.
*	Putting more than one dense layer is a successful choice, in fact the model with two dense layers of 128 neurons in the classifier reaches 0.87 accuracy.
*	The batch size affects how fast the accuracy would increase. In particular with smaller batch size, we had better accuracy with less epochs.

We also tested the following model:
* Learning rate 10^-4, batch size 8, one layer of 128 neurons, fine tuning freezed until 11

It is interesting because the first time we run the network it reached 0.91 validation accuracy. However, what we have not been able to reproduce the experiment. When we run again our code the accuracy couldn’t break the 0.34 barrier. There was a local minimum. We investigated and we found out a plausible explanation: the first time we trained our network, we didn’t reinitialize the seed (used to make experiment reproducible), so we had a different set of random weights initializing our neural network. Since a neural network is sensible to the weight initialization, the first time we were lucky enough to find a set of weights that made our net converge fast to a good solution. With this unfortunate experience, we understood how important is the weight initialization for a neural network.

## MobileNet V2

After some researches, we decided to have some trials also with MobileNet V2. All the experiments made with this typology of CNN reported almost the same results. All the validation accuracies were between 86% and 91%.
Firstly, we maintained along the experiments the same learning rate, because with this network it used to perform faster without loose performance. We initially varied the batch size and after tried 8, 16 and 32 we went for 16.
Then, we chose which weights of the network to freeze and which one to train. We tried two options:
1. 46 freezed (until block 4 of the network included)
2. 28 untrainable weights (until block 2 of the network included)

Then we started to “play” with the FC part of the whole network adding and removing different kinds of layers (Max Pool, Dropout and Dense).

The best result was obtained by the following set up and we reached 91% of accuracy (Kaggle’s submission):
* BS = 16, freezed layers of the CNN = 28, a 2D Max Pool layer (7, 7), 128 units in the first dense layer of the classifier, a dropout layer connected with the dense one (dropout = 0,5), LR = 10^-4
