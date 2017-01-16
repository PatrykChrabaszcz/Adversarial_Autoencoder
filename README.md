# Adversarial Autoencoder
University of Freiburg


Deep Learning Lab final project


Tensorflow implementation of [Adversarial Autoencoder](https://arxiv.org/abs/1511.05644).

![model](assets/model.png)

## Enviroment

- Python 3.5.2
- TensorFlow 0.12.1

(TODO: Add AAE description)

## Scenarios

In all experiments batch norm is used both in encoder and decoder network, discriminator
network does not use batch norm. Leaky ReLU with 0.2 leak was choosed as activation.

1. Autoencoder is trained on MNIST data using network with fully connected layers. 
Both encoder and decoder have two hidden layers (1000 neurons each). Discriminator 
has two layers (500 neurons each). Images are compressed to 5 dimensions. Gaussian 
normal distribution is used to sample from latent space.

2. Same as scenario 1 but we use convolutional neural network in both encoder 
and decoder. 

Encoder: 

    3x3x16 convolution
    4x4x32 convolution with stride 2
    4x4x64 convolution with stride 2
    7x7x5 convolution to reduce representation to 5 dimensions
    
Decoder:

    7x7x64 transposed convolution
    4x4x32 transposed convolution with stride 2
    4x4x16 transposed convolution with stride 2
    3x3x1  convolution

## References

- [DCGAN Tensorflow code](https://github.com/carpedm20/DCGAN-tensorflow)
- TODO: Add papers

## License

MIT License.
