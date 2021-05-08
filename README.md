# emotion-detection

Simple program that attempts to recognize facial expressions of people using a CNN.

## Dependencies

This project requires [PyTorch](https://pytorch.org/) and [OpenCV](https://docs.opencv.org/3.4/da/df6/tutorial_py_table_of_contents_setup.html).

## Databases

In order to train some neural networks, we used the following datasets:

* [FER2013](https://www.kaggle.com/deadskull7/fer2013)

## Approaches

Two approaches have been tried to solve the problem: one is to train convolutional neural networks from scratch, another is to fine-tune pretrained neural networks like `vgg` or `resnet`. 

### Learning From Scratch

### Transfer Learning

**I. Classifier warming up**

Since we have to change the classification part in the architecture of a pretrained neural network, we might want to train the freshly initialized classification layers to avoid a too strong modification of the weights in the feature part leading to a loss of the benefits of using a pretrained model for feature extraction. So in this first step, only the classifier layers are trainable.

| pretrained model | time | epochs | lr | momentum | max acc | min loss |
|-|-|-|-|-|-|-|
| vgg16 | 33m06s | 30 | 0.01 | 0.9 | 0.4032 | 1.549 |
| resnet | ... | ... | ... | ... | ... | ... |

**II. Fine-tuning**

Once the classification layers "warmed up", we fine-tune the whole network with a small learning rate to prevent modifying too much the feature extraction layers.

| pretrained model | time | epochs | lr | momentum | max acc | min loss |
|-|-|-|-|-|-|-|
| vgg16 | 80m44s | 100 | 0.0001 | 0.9 | 0.6124 | 1.048 |
| resnet | ... | ... | ... | ... | ... | ... |
