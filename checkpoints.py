import torchvision
import torch
import model_loader
from torchvision import datasets, transforms, models


def save_checkpoint(file_path, model, optimizer, model_name, class_to_idx):


    checkpoint = {
        'model_name': 'alexnet',
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'class_to_idx': class_to_idx,
        }

    return torch.save(checkpoint, file_path)



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model_name = checkpoint['model_name']
    classifier = checkpoint['classifier']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']

    model = getattr(models, model_name)(pretrained=True)
    model.classifier = classifier
    model.load_state_dict(state_dict)
    return model, class_to_idx
