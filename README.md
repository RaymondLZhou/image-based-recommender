# Artistic Learning

Neural style transfer built on TensorFlow and Keras in Python with a deep convolutional neural network (CNN). Creates artistic image by recombining input content and style, using NumPy to shape neural representations.

## Description

Performs neural style transfer as described in the original paper, [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf) by Gatys, Ecker, and Bethge. The following is taken from the paper.

We introduce an artificial system based on a deep neural network that creates artistic images of high perceptual quality. The system uses neural representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for the creation of artistic images.

Each layer of a convolutional neural network can be understood as a collection of image filters, each of which extracts a certain feature from the input image. When convolutional neural networks are trained on object recognition, they develop a representation of the image that makes object information increasingly explicit along the processing hierarchy.

The representations of content and style in the convolutional neural network are separable. That is, we can manipulate both representations independently to produce new, perceptually meaningful images. To demonstrate this finding, we generate images that mix the content and style representation from two different source images.

## Examples
![output1](images/output1.png)

![output2](images/output2.png)

![output3](images/output3.png)

![output](images/output.png)

## Getting Started

How to run the application

1. Clone the repository.
2. Open [hyperparameters.py](src/hyperparameters.py) and adjust the hyperparameters as desired. 
3. Run [main.py](src/main.py) with ```python main.py```. The output image is saved in [data/output](data/output).

## Built With

* [TensorFlow](https://www.tensorflow.org/) - Deep Learning

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
