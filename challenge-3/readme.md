# APPROACH

Let’s start to define how we approached the problem. 
After long research on the internet, we found different papers explaining ways to solve the VQA problems. We decided to apply one of the simplest ones due to time management (we were in the middle of the exams session and we couldn’t use all our time for the project, even if we would have liked it!).
The way chosen is the one named “VIS+LSTM”. We made use of LSTM for managing the memory of the words of the questions, and we treated the image like if it was a part of the sentence. 
We embedded the questions (GloVe) and we also encoded the images (VGG). Finally, we concatenated the encoded text and the features extracted from the pictures.

Something like the following graph:

![alt text](https://github.com/leomandru/AN2DL-competitions/blob/main/challenge-3/graphics/img1.png)

In this way, we can look at the problem almost as to a classification one.

# DATASET

You can download the dataset from this [link](https://www.kaggle.com/c/anndl-2020-vqa).

## FROM JSON TO OUR MODEL

Firstly, we did not divide the images into 3 folders for training, validation, and test. We kept all the pictures in the “Images” directory and we used dataframes (pandas) for managing the separation of the data:
* 1 dataframe starting from “train_questions_annotations.json” and then divided into 2 parts for training and validation 
* 1 dataframe starting from “test_question.json” for the test dataset

## CUSTOM GENERATORS

The dataset is not just made up of images. There are also questions and in the training phase also answers.

For managing everything we designed 2 custom generators extending the class `Keras.layers.Sequence`:

* the first one is “CustomDataset” for training and validation data
*	the second is “TestCustomDataset” for test data

They must be different, in fact, the `__getitem__()` method has to return slightly different things. In the first case questions, images, and answers grouped in batches, while for the second one only questions and pictures one after the other (no batches).

The question and the answers are not returned as strings. Before their insertion into the custom datasets, we pre-processed them.


## QUESTIONS, ANSWERS, and IMAGES PRE-PROCESSING

### Questions

In general, as we have anticipated, we used GloVe 42B 300d (truncated to 300K words) for the text embedding. So, the questions were initially tokenized exploiting the word index `word_idx` obtained from the GloVe dictionary.

### Answers

We used in the project one-hot encoding and categorical cross-entropy. So, we had to transform all the answers (categories) into NumPy arrays made of 0s and 1s. 

In the end, we had 5 matrixes:


1. **Training questions matrix** → a matrix with the following dimension : number of training questions (80% of the total questions) x 23 (longest question present in the dataset because we used padding)
2. **Training answer matrix** → a matrix with the following dimension : number of training answers (80% of the total answers) x 58 (number of possible answerscategories)
3. **Validation questions matrix** → a matrix with the following dimension : number of validation questions (20% of the total questions) x 23 (longest question present in the dataset because we used padding)
4. **Validation answer matrix** → a matrix with the following dimension : number of validation answers (20% of the total answers) x 58 (number of possible answerscategories)
5. **Test questions matrix** → a matrix with the following dimension : number of test questions x 23 (longest question present in the dataset because we used padding)

### Images

The only preprocessing operation done on the images was the resize to 224x224 (allowed input for the VGG19 for feature extraction).

## EMBEDDING

As we’ve already mentioned, we used GloVe to embed questions. We followed the classical approach for creating the word index and the embedding matrix. We used the words' index for tokenizing the question and the embedding matrix as the values of the weights of the embedding layer of the model.

We struggle a lot with understanding all the mechanisms related to the embedding matrix, but in the end, we have deeply understood how it works. 

For a matter of memory and also in terms of utility for this specific problem, we truncated the embedding matrix to the most used 300K words. So, we had to use a specific “empty” token in case of those words not present in the embedded matrix (consequently also in the words' index).

## FEATURE EXTRACTION

As for the feature extraction from the images, we inserted before our real model a VGG19 pre-trained on the Imagenet dataset. We then concatenated the extracted features of the encoded text to make the classification and obtain the predicted answers.

# MODEL TRIALS

1. The simplest trial (*following figure*):

* TRAINING INPUT → question, image, answer
* TEST INPUT → question, image
* FEATURE EXTRACTION → VGG19 (Imagenet), dense
* EMBEDDING → GloVe 42B 300d (truncated to 300K words)
* TEXT LAYERS → embedding, 2 layers of LSTM (no bidirectional), dense
* FC PART → 2 dense layers (1024 neurons and 58 neurons)

![alt text](https://github.com/leomandru/AN2DL-competitions/blob/main/challenge-3/graphics/img2.png)

2. We tried to play with the position on the image feature extraction part of the model. We’ve seen in some papers how the position is fine-tuned by the designer of the net. Someone used to put it as the first word of the sentence, someone else as the last word of the question. We tried to put the picture at the beginning and at the end in this way (*following figure*) and we obtained better results:

* TRAINING INPUT → image, question, image, answer
* TEST INPUT → image, question, image
* FEATURE EXTRACTION → VGG19 (Imagenet), dense
* EMBEDDING → GloVe 42B 300d (truncated to 300K words)
* TEXT LAYERS → embedding, 2 layers of LSTM (bidirectional), dense
* FC PART → 2 dense layers (1536 neurons and 58 neurons)

![alt text](https://github.com/leomandru/AN2DL-competitions/blob/main/challenge-3/graphics/img3.png)
