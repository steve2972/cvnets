# CVNets Repository

This repository contains several easy-to-use convolutional networks for research/academic purposes.
The list of currently available models include:

- MobileNet v3
- MobileVIT
- MobileVIT v2

Work in progress:
- MobileNet versions 1,2
- EfficientNet
- Swin Tranformers
- VIT
- ResNet
- and many more

## How to train

This repository currently uses PyTorch Lightning to train the models. I am currently trying to find good initial parameters for adequate model training. The training parameters are currently set at:

- optimizer: stochastic gradient descent. Note that although adam converges faster, the generalization ability is weaker than sgd.
- learning rate: 0.256
- scheduler: stepped learning rate. The learning rate is multipled by a decay parameter (0.97) every 2 epochs.