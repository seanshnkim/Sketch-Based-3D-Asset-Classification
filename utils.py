import torch
from torchvision import datasets, models, transforms
import pandas as pd
# import time
from matplotlib import pyplot as plt
import os
import cv2
# from torchvision import transforms as T
from torch import nn
from tqdm import tqdm

def load_efficientnet_v2_model():
    # Load the pre-trained EfficientNetV2-L model
    # effNet_v2_large = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
    effNet_v2_large = models.efficientnet_v2_l()
    # load the pretrained weights
    effNet_v2_large.load_state_dict(torch.load('checkpoints/efficientnet_v2_l-59c71312.pth'))
    effNet_v2_large.eval()
    return effNet_v2_large

def load_efficientnet_v1_model():
    # Load the pre-trained EfficientNetV2-L model
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model

def load_resnet50_model():
    """
    Load the pre-trained ResNet50 model.
    """
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def preprocess_efficientnet_v1():
    """
    Define the transform for the input image/frames.
    Resize, crop, convert to tensor, and apply ImageNet normalization stats.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),])
    return transform

# NOTE : This function is used for efficient_v2
def preprocess_efficientnet_v2():
    """
    Define the transform for the input image/frames.
    Resize, crop, convert to tensor, and apply ImageNet normalization stats.
    """
    # transform =  transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                          std=[0.229, 0.224, 0.225]),])

    return transforms.Compose([transforms.ToTensor(), models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()])

def read_classes():
    """
    Load the ImageNet class names.
    """
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def read_classes_customized():
    """
    Load the ImageNet class names.
    """
    class_list_df = pd.read_csv("class_list.csv") 
    return list(class_list_df['class_name'])


def run_through_model(model, val_images_dir, val_ground_truth, n_data, DEVICE='cuda', transform=None):
    with open(val_ground_truth, "r") as f:
        ground_truth = f.readlines()
    
    top5_cnt = 0
    idx_data = 0
    
    with torch.no_grad():
        for idx_data in tqdm(range(n_data)):
            for (root, dirs, files) in os.walk(val_images_dir):
                files.sort()
                file_path = os.path.join(root, files[idx_data])
                
                image = cv2.imread(file_path)
                input_tensor = transform(image)
                input_batch = input_tensor.unsqueeze(0)
                input_batch = input_batch.to(DEVICE)
                
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 5)
                top5_catid_list = top5_catid.tolist()
                
                curr_gth_int = int(ground_truth[idx_data].split()[1].rstrip('\n'))
                if curr_gth_int in top5_catid_list:
                    top5_cnt += 1
                    
                # idx_data += 1
            
    accuracy = round(top5_cnt / n_data, 4)
    
    return accuracy

def plot_time_vs_iter(model_names, time_lists, device):
    """
    Plots the iteration vs time graph for given model.
    :param model_name: List of strings, name of both the models.
    :param time_list: List of lists, containing time take for each iteration
        for each model.
    :param device: Computation device.
    """
    colors = ['green', 'red']
    plt.figure(figsize=(10, 7))
    for i, name in enumerate(model_names):
        plt.plot(
            time_lists[i], color=colors[i], linestyle='-', 
            label=f"time taken (ms) {name}"
        )
    plt.xlabel('Iterations')
    plt.ylabel('Time Taken (ms)')
    plt.legend()
    plt.savefig(f"outputs/time_vs_iterations_{device}.png")
    plt.show()
    plt.close()