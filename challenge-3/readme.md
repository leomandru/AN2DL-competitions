# APPROACH

Let’s start to define how we approached the problem. 
After long research on the internet, we found different papers explaining ways to solve the VQA problems. We decided to apply one of the simplest ones due to time management (we were in the middle of the exams session and we couldn’t use all our time for the project, even if we would have liked it!).
The way chosen is the one named “VIS+LSTM”. We made use of LSTM for managing the memory of the words of the questions, and we treated the image like if it was a part of the sentence. 
We embedded the questions (GloVe) and we also encoded the images (VGG). Finally, we concatenated the encoded text and the features extracted from the pictures.

Something like the following graph:

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

In this way, we can look at the problem almost as to a classification one.

# DATASET

## FROM JSON TO OUR MODEL

Firstly, we did not divide the images into 3 folders for training, validation, and test. We kept all the pictures in the “Images” directory and we used dataframes (pandas) for managing the separation of the data:
* 1 dataframe starting from “train_questions_annotations.json” and then divided into 2 parts for training and validation 
* 1 dataframe starting from “test_question.json” for the test dataset

## CUSTOM GENERATORS

The dataset is not just made up of images. There are also questions and in the training phase also answers.

For managing everything we designed 2 custom generators extending the class Keras.layers.Sequence:

•	the first one is “CustomDataset” for training and validation data
•	the second is “TestCustomDataset” for test data

They must be different, in fact, the “__getitem__()” method has to return slightly different things. In the first case questions, images, and answers grouped in batches, while for the second one only questions and pictures one after the other (no batches).

The question and the answers are not returned as strings obviously. Before their insertion into the custom datasets, we pre-processed them.


## QUESTIONS, ANSWERS, and IMAGES PRE-PROCESSING

### Questions

### Answers

### Images

## EMBEDDING

## FEATURE EXTRACTION

# MODEL TRIALS

