# Image_Classification_Flowers

This is my codebase for Flower's image classification Deep Learning project with Udacity

In this project, I train an image classifier to recognize 102 different species of flowers. This can be embedded within an application


There are five (5) files, two main files (train.py and predict.py) and three helper files

- data_loader.py: loads train data, test data, and validation data with torchvision's ImageFolder
- model_loader.py: loads a pretrained network such as AlexNet from torchvision.models and the parameters are frozen
- checkpoints.py: saves the trained model as a checkpoint along with associated hyperparameters and the class_to_idx dictionary. This module also loads a saved checkpoint and rebuilds the model
- train.py: trains the feedforward classifier, while the parameters of the feature network are left static
- predict.py. takes the path to an input mage and a checkpoint, then returns the top K most probably labels for that image



To run this application on the command line, there are required and optional arguments for train.py and predict.py

#### train.py
- data_dir. directory with train, test, and validation subfolders. Mandatory argument. - 'str'
- --save_dir. directory to save trained model_loader. Optional argument with. - 'str'
- --arch. Model architecture e.g alexnet, vgg13, and vgg16. Optional argument used if provided else Alexnet will be used with - 'str'
- --learning_rate. optional argument. - 'float'
- --hidden_units: specifies the number of units in the hidden layer. optional argument used if provided else 1024 is use. - 'int'
- --epochs: 'Number of epochs', type = int
- --gpu: specifies gpu or cpu is used in training the model. Optional argument

#### predict.py
- image_dir: path to input image. Required argument. - 'str'
- load_dir: path to checkpoint. Required argument. - 'str'
- --top_k: specifies the top likely classes. Optional argument. - 'int'
- --category_names: helps to map classes to actual flower categories. Optional argument. - 'str'
- --gpu: specifies gpu or cpu is used in predicting the label of the input image. Optional argument. - boolean
