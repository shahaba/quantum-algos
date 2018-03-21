# Autoencoders: From Classical to Quantum 

Classical autoencoders are neural networks that can reduce the dimensionality of data, effectively compressing an input from a high dimensional representation to a low dimensional encoding. We can approach this problem by training multilayered networks where at each layer we can chose to represent the data on a smaller dimension. 


This model has inspired a group at Harvard University to introduec a quantum version of the autoencoder in order to compress a particular dataset of quantum states. A new model is necessary as classical compression algorithms can't be used. Their implementation is a quantum-classical hybrid, where the simulation and the data would sit on a quantum device, the parameters of the autoencoder can be optimized using classical techniques.

The quantum autoencoder encodes a the Hilbert space as follows:
