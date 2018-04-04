Quantum Algorithms
==================


## Project 1: Autoencoders: From Classical to Quantum 

Classical autoencoders are feed-forward neural networks that can reduce the dimensionality of data, effectively compressing an input from a high dimensional representation to a low dimensional encoding. Hinton approached this problem by training multilayered networks, where at each layer the data is represented on a smaller dimension. 

In more technical terms, a classical autoencoder will optimize its parameters across a given training set, (n + k)-bit input string x, and attempt to reconstruct that input after erasing k bits during the process. If the autoencoder can reproduce the input with some accuracy, the remaining n-bits (refered to as the latent space) represent the compressed encoding of x. Therefore, we compress the neural network such that the encoded information is a close match to the training set.

This model was simplified, and translated into its quantum version by the As-Gu group at Harvard University. Their quantum autoencoder is designed to compress a particular dataset of quantum states. Their implementation falls under a quantum-classical hybrid, where the simulation and the data would sit on a quantum device, but the parameters of the autoencoder would be optimized using classical techniques.


A single iteration of the algorithm can be broken down into the following steps:

1. Prepare the input, and the reference state. The preparations are assumed to be efficient
2. Evolve the input under the encoding unitary, where p is the set of parameters at a given optimization step
3. Measure the fidelity between the trash state and the reference state via a SWAP test
4. Apply cost function to the classical optimization routine and return a new set of parameters for the encoding unitary.
