import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
import logging
import checkpoints
import json
import sys
import os
import model_loader
from PIL import Image
from torchvision import transforms
import argparse
from workspace_utils import active_session


logging.basicConfig(
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def preprocess_img(img_dir):
    img_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                     ])
    pil_img = Image.open(img_dir)
    img = img_transform(pil_img)
    return img


def class_pred(args, model, class_to_idx, device, top_k=1):
    if args.top_k:
        top_k = args.top_k

    #convert img to pil
    img_dir = args.image_dir
    with torch.no_grad():
        model.eval()
        img = preprocess_img(img_dir)
        img = img.unsqueeze_(0) #Add an axis for feed forward
        model, img = model.to(device), img.to(device)
        logits = model.forward(img)
        ps = torch.exp(logits)
        inv_map = {v: k for k, v in class_to_idx.items()} #Maps class to string index which is passed into category json
        top_p, top_class = ps.topk(top_k, dim=1)
        top_p = top_p.numpy()[0]
        top_class = top_class.numpy()[0]
        #print(top_p, top_class)
        top_idx = [inv_map[i.item()] for i in top_class]

        return top_p, top_idx



def label_map(args, top_idx, mapping_dir='cat_to_name.json'):
    if args.category_names:
        mapping_dir = args.category_names

    with open(mapping_dir, 'r') as f:
        cat_to_name = json.load(f)
        labels = [cat_to_name[i] for i in top_idx]
    return labels


def plot_predictions(args, model, class_to_idx, device, topk=5):
    top_p, top_idx = class_pred(args, model, class_to_idx, device, topk)
    predicted_names = label_map(args, top_idx)
    top_label = predicted_names[np.argmax(top_p)]

    px = 1/plt.rcParams['figure.dpi']
    fig, axs = plt.subplots(2, 1, figsize=(224*2*px, 224*2*px))
    axs[0].axis('off')
    axs[0].set_title(top_label)
    imshow(process_image(image_path), ax = axs[0])

    axs[1].barh(predicted_names, width = top_p)
    plt.tight_layout()



if __name__ == "__main__":

    logging.info('loading argument into argspace')

    my_parser = argparse.ArgumentParser(description='Arguments for Predicting Images')
    my_parser.add_argument('image_dir', action='store', type=str)
    my_parser.add_argument('checkpoint_dir', action='store', type=str)
    my_parser.add_argument('--top_k', action='store', type=int)
    my_parser.add_argument('--category_names', action='store', type=str)
    my_parser.add_argument('--gpu', action='store_true', default=False)


    args = my_parser.parse_args()


    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    with active_session():

        logging.info(f"loading checkpoint {args.checkpoint_dir}")
        model, class_to_idx = checkpoints.load_checkpoint(args.checkpoint_dir)
        model.eval()

        logging.info(f"predicting image input from {args.image_dir}")
        top_p, top_idx = class_pred(args, model, class_to_idx, device)

        labels = label_map(args, top_idx)
        formatted_ps = [str(round(i * 100, 2))+'%' for i in top_p]
        print(f"prediction: {list(zip(labels, formatted_ps))}")
        #plot_predictions(args, model, class_to_idx, device, topk=5)
