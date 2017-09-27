# TensorFlow

## TensorFlow/tutorials

  Tutorial for Tensorflow

  Modified the python files from http://tensorflow.org to ipynb files


## TensorFlow/cifar10
CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN).
Deep layers and structures are both important.

### cifar10_35.ipynb
Demostrate a base model, the test accuracy is about 35%

### cifar10_one_layer.ipynb
Demonstrate svm and softmax classifier with tensorflow, the test accuracy is about 35%

### cifar10_two_layer.ipynb
Demonstrate two-layer full connnected net with tensorflow, the test accuracy is about 50%

### cifar10_fully_connected_net.ipynb
Fully connected deep net for any number of layers, for any dropout, for any batch norm, for any L2 regularization
Demonstrate about 55% test accuracy of four-layer net on cifar10 dataset;
Demonstrate about 52% and 53% test accuracy of eight-layer net on cifar10 dataset.

### cifar10_100_by_bug.ipynb
bug from tf.nn.in_top_k(logits, labels, 1) 
The bug makes the test accuracy is 100%

### cifar10_65.ipynb
Demonstrate about 65% test accuarcy for a net like "conv - RELU - POOL - FC - RELU - SOFTMAX" on cifar10 dataset

### cifar10_80.ipynb
Demonstrate about 80% test accuarcy for a net like "conv-RELU-POOL - conv－RELU-POOL -conv-RELU-conv-RELU-POOL -FC-RELU-FC-RELU - SOFTMAX" on cifar10 dataset


# SVHN

## TensorFlow/SVHN

  Demonstrate a unified approach that integrates localization, segmentation, and recognition steps 
  via the use of deep convolutional neural network that operates directly on the image pixels.
  Algorithms demo, refer to the paper:
  Ian, J. G, "multi-digit number recognition from street view imagery using deep convolutional 
  neural networks.
  
 ### SVHN.ipynb
Demonstrate about 97% validation accuarcy on SVHN set (not including extra-data)
 
