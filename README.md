# [doodle-bot](https://github.com/messiest/doodle-bot)
## Conditional generative adversarial netrwork used for image generation.

### Author: [Christopher Messier](messiest.github.io/)
##### email: [messier.development@gmail.com]()

### Summary
Prior to working on this project, I became interested in the idea of hierarchical concepts.
Think about a breed of dog.
No matter what kind of dog you thought of, you thought of an instance of a dog.
Whether it's a bulldog, corgi, german shepard, or chihuahua, they are all kinds of dogs.
That means when identifying a type of dog, it's also important to identify that it is a dog.
This is something that is often missing in the application of convolutional neural networks.

Convolutional neural networks alone lack the ability to provide contextual meaning to an image.
They might be quite good identifying a certain class, but they are unable to recognize that this is an instance of some super class of objects; dogs $\to$ corgis, if you will.
To arrive at an understanding of theses hierarchical concepts, I'm employing generative adversarial networks.
These models have interactions between two networks, a _generator_, and a _discriminator_.

The _discriminator_ is a convolutional neural network that is used for image classification.
As the _discriminator_ is trained on class images, it is also fed random noise.
This random noise is output from the _generator_ network.
The classification that is returned from the _discriminator_ is used by the _generator_ to update the noise that is being output, to better match the true class.
In doing this, you are training a model to not only recognize the object, but be able to generate __original images that correspond to the training data classes__.

As an early proof of concept, I implemented this model on the generation of "hand-written" digits that were trained using the MNIST data set.

#### Tasks
1. Image importing and writing to S3.
2. Sampling from image-net.org and writing to S3.
3. Sampling existing images from S3 and writing to disk.
4. Build GAN model.
5. Run on MNIST dataset.

Generated Output Example:

#### MNIST
![alt text](exmples/example_1.png "Logo Title Text 1")


#### [WordNet](http://www.nltk.org/howto/wordnet.html)
Symmantic database of words. Organizes words into hierarchies using _synsets_, or collections of words with related meaning.
This is accessed via the `nltk.corpus` library.


#### [ImageNet](https://image-net.org)
Online image data base, with a structure based on WordNet.
This is used for searching for, and downloading of images for training.


#### Software Model
`doodle-bots`'s functionality is spread across numerous repositories.
In each you will find numerous modules and libraries that provide the functionality.
This directory, doodle-bot-learn, provides the image generation and classification modules written with TensorFlow.

