# GAN
Generative Adversarial Nets

This repo has the purpose of implementing GANs in Python.

1) GAN_1D.py / GAN_1D-example.ipynb:
The file contains a toy example where a GAN is used to model a 1-dimensinal Gaussian distribution.
Both the generator and the discriminator is a MLP and mini-batches are used. The example alternates between updating the discriminator and generator, one update each, but this can easily be changed in the code such that the discriminator is updated several times before the generator gets an update.

2) GAN-2D.ipybn:
Contains a 2D GAN on the MNIST data set using Keras. When running the code it often happens that the generator fails to learn anything, hence producing pure noise pictures. This issue is fixed in the Wasserstein GAN.

3) WGAN.py:
This is the implementation from the original paper on Wasserstein GANs. I have chosen to use CIFAR-10 benchmark data set because I like cats and dogs.. however not frogs, well the former outweights the latter. 
The implementation is done in PyTorch.

4) WGAN_training.gif:
A GIF illustrating the generator's ability to generate life-like pictures from the same noise during training. The pictures look more and more realistic. This was done using the code in WGAN.py on an Amazon cloud computing GPU for 100 iterations of training CIFAR-10.
