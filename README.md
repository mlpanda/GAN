# GAN
Generative Adversarial Nets

This repo has the purpose of implementing GANs in Python.

The first file upload is "GAN_1D.py" which contains a toy example where a GAN is used to model a 1-dimensinal Gaussian distribution.
Both the generator and the discriminator is a MLP and mini-batches are used. The example alternates between updating the discriminator and generator, one update each, but this can easily be changed in the code such that the discriminator is updated several times before the generator gets an update.
