# DATASET

We wrote some pieces of code to create training and validation sets. Thus, we obtained 2 folders (training and validation) containing 80% and 20% of the whole images set (all the images from the 4 teams belonging to the 2 different crops).

# FIRST APPROACH

We tried to readapt the Multiclass Segmentation Notebook (written during classes) to solve the new assigned problem. 
We initially focused on the CustomDataset class. We made some modifications to manage the new dataset (different from the one of the lab). 
With the same model of the laboratory, we obtained about 15% meanIoU accuracy (HW2_01.ipynb).

# SECOND APPROACH
## DATA and MODEL

To find a solution in order to obtain better results, we started looking both at the data and our model. We noticed some peculiarities:
1.	Different teams have taken pictures with diverse proportions/resolutions
2.	The spatial resolution of our model was 256x256
3.	To submit our prediction, we had to resize through bilinear interpolation the images from 256x256 to their original size

## CROPPING

After these considerations, we decided to implement a method for cropping each image of the dataset into small pieces of 256X256 spatial resolution. We then used these portions of the pictures to train the model. 
To predict image classes, we used to divide it into small pieces, process them through the neural network, and recompose them into the segmented image.

### Different spatial resolutions

As we said before, one problem we had to face was the fact that not all the images have a resolution multiple of 256X256. We had special care dealing with these images (we did not simply resized them).
Adopting this method resulted in a quite successful prediction since we managed to increase our prediction to about 50% (HW2_02.ipynb).

# THIRD APPROACH
We also tried to solve the problem with the approach described in the Binary Segmentation notebook. So, without a CustomDataset, but with generators.
We properly modified code to deal with the new type of problem, but we initially obtained awful results (almost 0% of meanIoU, PROBABILYprobably due to some errors). 
Applying then the cropping approach described before, we solved the problem. In fact, we reached approximately the same results obtained with the CustomDataset, 45% (HW2_03.ipynb).

# U-NET
The next and final step was to look on the internet at the various implementations of U-NET and include it in our code. After training a U-NET model for 10 epochs, we got to 60% meanIOU accuracy (HW2_Final.ipynb). With this method, we have the U-NET model power aligned with the efficiency of having a Dataset properly structured for it.
Furthermore, the peculiarity of this network is the skip connections usage. It has the aim to preserve the most useful information for segmenting the images. The model we found online uses batch normalization layers for normalizing data. Moreover, it is believed that it speeds up the training of the neural network. The only issue with using batch normalization is that, experimentally, it doesn’t work so well when the batch size is too small or when its dimension varies during the training. For this reason, we chose a batch size of 64 that is reasonable since we operate with a much bigger dataset comparing the original one. 

# CONSIDERATIONS

## LEARNING RATE
We started the project with the learning rate equal to 1e^-5, but we have immediately noticed it was inappropriate. In fact, the first time we’ve run the notebook, it was stuck at almost 0% of meanIoU. Changing it to 1e^-4 we solved that problem. Thus, we kept it like this for all the notebooks

## BATCH SIZE
When we first approached the problem, we used to have a small number of images since we chose for a resizing approach and not for the cropping one. Thus, we initially set the batch size to 8/16. With the cropping approach, the dataset went from a size of about 730 images to more than 85000. It was therefore impossible to continue using a small batch size like 8/16 (also due to the problem described before of the batch normalization with the U-NET). Therefore, we then changed it to 64. We couldn’t opt for a bigger one because it would have been impossible to upload on the GPU data when we were using UNET. 
