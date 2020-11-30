# MAIS_202 - FaceGen
McGill Artificial Intelligence Society Bootcamp MAIS202 - Final Project

Training data retrieved from [Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset).

## Project Description
The aim of this project is to utilize a Generative Adversarial Network to create realistic photos of human faces by training the model on the CelebA dataset of celebrity faces.

My project seeks to take an elementary approach to Image Generation by using the readily available CelebA dataset of celebrity faces. The GAN model is comprised of a basic discriminator which can tell the difference between real and fake images and a generator which seeks to keep improving and generate more realistic photos to trick the discriminator.

## To run the project
Download the eight files:
```
Image_GAN_Model.py
Predict.py
generator_weights_200.h5
dataset_processed_25000.part1.rar
dataset_processed_25000.part2.rar
dataset_processed_25000.part3.rar
dataset_processed_25000.part4.rar
dataset_processed_25000.part5.rar
```
Unzip the 5 .rar files to get a single dataset_processed_25000.npy file.
Make sure to put all files in the same folder.

To train the model from beginning to end, run 
```
Image_GAN_Model.py
```

This will generate faces every 10 epochs and save them as an image as well as give statistics about the model.

The training time takes about 4-5 hours.

Note that the Image_GAN_Model.py file used to be a combination of two separate files: one for preprocessing the image data, and one for creating and training the GAN model.

Therefore, the code for preprocessing has been commented out.

To simply test the trained model, run
```
Predict.py
```
This will generate 100 faces based on the weights of the generate after 200 epochs of training and save it to an image file.

## Repository Organization

This repository contains the scripts used to train and test the model as well as a poster / slideshow of the final project.

1. Deliverables/
	  * deliverables submitted to the MAIS Intro to ML Bootcamp organizers
  
2. Final Project/
    * [Sample Results] contains the generated images at every 5 epochs of training from 0 to 200 as well as the graphs for the statistics on the model.
    * [Image_GAN_Model.py] contains the code for preprocessing the image data and training the GAN model.
    * [Predict.py] contains the code for testing the model on the weights at 200 epochs of training.
    * [dataset_processed_25000.npy] is the dataset of 25000 images in numpy format used by [Image_GAN_Model.py].
    * [generator_weights_200.h5] is the file containing the information and weights of the model at 200 epochs in h5 format used my [Predict.py]
    * [MAIS202 Final Project Poster Slideshow.pptx] is the slideshow/poster file containing a detailed overview of the overall project.
