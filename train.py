import pandas as pd
import numpy as np
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import sys
import os
import logging

import data_loader
import model_loader
import checkpoints
from workspace_utils import active_session


logging.basicConfig(
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)



def validation(model, validation_data, criterion, device):
    validation_loss = 0
    validation_accuracy = 0
    for images, labels in validation_data:
        model = model.to(device)
        images = images.to(device)
        labels = labels.to(device)
        logits = model.forward(images)
        loss = criterion(logits, labels)
        validation_loss += loss.item()

        ps = torch.exp(logits)
        top_prob, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        validation_accuracy += accuracy


    return validation_loss, validation_accuracy




def train_model(model, train_data, validation_data, device, learning_rate=0.001, epochs=10, print_every=40):

    if args.learning_rate:
        learning_rate = args.learning_rate

    if args.epochs:
        epochs = args.epochs

    criterion = nn.NLLLoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    train_loss = []
    validation_loss_list = []
    steps = 0
    for epoch in range(epochs):
        running_loss = 0
        running_accuracy = 0

        for images, labels in train_data:
            #print(images.shape)
            steps += 1
            model = model.to(device)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() #Zero out the gradients
            #images = images.view(64, 3, 50176) #flatten_tensor(images)

            logits = model.forward(images)
            loss = criterion(logits, labels)

            loss.backward() #Backpropagation
            optimizer.step() #update weights

            ps = torch.exp(logits)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            running_accuracy += torch.mean(equals.type(torch.FloatTensor))
            running_loss += loss.item()



            if steps % print_every ==0:#Print losses every print_every
                #print(steps)
                model.eval()
                with torch.no_grad():
                    validation_loss, validation_accuracy = validation(model, validation_data, criterion, device)
                validation_loss_list.append(validation_loss)

                train_loss.append(running_loss)


                model.train()
                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(validation_loss/len(validation_data)),
                  "Valid Accuracy: {:.3f}%".format(validation_accuracy/len(validation_data)*100))

    return model, optimizer



if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description='Arguments for Training Images')

    my_parser.add_argument('data_dir', action='store', type=str)
    my_parser.add_argument('--save_dir', action='store', type=str)
    my_parser.add_argument('--arch', action='store', type=str)
    my_parser.add_argument('--learning_rate', action='store', type=float)
    my_parser.add_argument('--hidden_units', action='store', type=int)
    my_parser.add_argument('--gpu', action='store_true', default=False)
    my_parser.add_argument('--epochs', action='store', type=int)

    args = my_parser.parse_args()

    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'



    with active_session():
        start = pd.Timestamp.now()

        logging.info("Loading Deep Learning Model...")
        model, model_name, layers = model_loader.load_model(args, train=True)

        logging.info(f"Loading {args.data_dir} data")
        train_data, validation_data, test_data = data_loader.loader(args.data_dir)

        logging.info(f"Training {model_name} Deep Learning Network...")
        model, optimizer = train_model(model, train_data, validation_data, device)

        logging.info(f"{model_name} Model Training Completed")
        logging.info(f"Training process took {pd.Timestamp.now() - start}")
        if args.save_dir:
            logging.info("Saving trained model")
            checkpoints.save_checkpoint(args.save_dir, model, optimizer, model_name, train_data.dataset.class_to_idx)

        logging.info("process completed")
