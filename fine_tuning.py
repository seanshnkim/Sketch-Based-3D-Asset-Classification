from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from PIL import Image
import torch
from tqdm import tqdm
import os
import csv
import logging 
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

imgnet_class_txt_path = '/home/ubuntu/Sketch-Recommendation/data/imagenet_classes.txt'
val_data_path = '/home/ubuntu/Sketch-Recommendation/data/val.txt'

transform_test = transforms.Compose([transforms.ToTensor(), 
                        models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()])

def find_classes(class_fPath):
    classes_dict = {}
    with open(class_fPath, 'r') as f:
        for s in f.readlines():            
            class_id, class_name = s.split(', ')
            classes_dict[class_id] = class_name
    return classes_dict

def make_dataset(data_fPath, class_dict): # val.txt
    images, labels = [], []
    with open(data_fPath, 'r') as f:
        for s in f.readlines():
            image_path, label_num = s.split(' ')
            images.append(image_path)
            # labels.append(class_dict[label_num.rstrip('\n')])
            labels.append(int(label_num.rstrip('\n')))
    return images, labels
        
class ImageDataset(Dataset):
    def __init__(self, data_fPath, transform=None, train=False):
        # self.imgs = list(sorted(os.listdir(self.path)))
        self.transform = transform
        self.class_dict = find_classes(class_fPath=imgnet_class_txt_path)
        self.images, self.labels = make_dataset(data_fPath, self.class_dict)
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images) 
    
    def __getitem__(self, idx):
        image_path = os.path.join('/home/ubuntu/Sketch-Recommendation/data/ImageNet_data/ImageNet_val', self.images[idx])
        image = Image.open(image_path).convert('RGB')
        # self.transform(image) => torch.Size([3, 480, 480])
        image_trs = self.transform(image)
        return image_trs, self.labels[idx]




# TODO: 빼기
class ImageDataLoader(DataLoader):
    def __init__(self):
        
        # # batch: [data1, data2, ..., dataN]
        # # 사용하는 경우: 1) data size가 batch 내에서 다이나믹한 경우 2) 토치 텐서가 아니거나, 에러가 나거나
        # def collate_fn(batch):
        #     batch = torch.stack([torch.FloatTensor(b) for b in batch])
        #     return batch
        
        # transform using for EfficientNet_v2
        transform_val = transforms.Compose([transforms.ToTensor(), 
                        models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()])
        
        val_dataset = ImageDataset(data_fPath=val_data_path, transform=transform_val)
        self.val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
    def get_data_loader(self):
        return self.val_loader
    

def load_efficientnet_v2_model(model_param_path):
    # Load the pre-trained EfficientNetV2-L model
    # effNet_v2_large = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
    effNet_v2_large = models.efficientnet_v2_l()
    # load the pretrained weights
    effNet_v2_large.load_state_dict(torch.load(model_param_path))
    effNet_v2_large.eval()
    return effNet_v2_large


def validate(model_param_path, logger):
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
    else:
        device = torch.device("cpu")
    
    # load the pre-trained EfficientNetV2-L model
    effNet_v2_large = load_efficientnet_v2_model(model_param_path).to(device)
    
    # load the dataloader
    data_loader = ImageDataLoader()
    val_loader = data_loader.get_data_loader()
    
    # validate top-1 & top-5 accuracy 
    cnt_top_1 = 0
    # top_5_correct = 0
    total = 0
    correct = 0
    
    with torch.no_grad():
        for data in tqdm(val_loader):
            # images.shape = torch.Size([10, 3, 480, 480])
            # type(labels) = <class 'tuple'>
            images, labels = data
            images = images.to(device)
            outputs = effNet_v2_large(images)
            
            # top-1 accuracy
            predicted = torch.argmax(outputs.data, 1)
            total += len(labels)
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    cnt_top_1 += 1
            if total % 100 == 0:
                logger.info(f'{total}th image: {cnt_top_1 / total}')
            # top-5 accuracy
            # predicted = outputs.topk(5, 1, largest=True, sorted=True)[1]

if __name__ == "__main__":
    effnet_v2_logger = logging.getLogger(__name__)
    effnet_v2_logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler('effnet_v2.log')
    effnet_v2_logger.addHandler(file_handler)
    
    ch = logging.StreamHandler()
    effnet_v2_logger.addHandler(ch)
    
    effnet_v2_logger.info('top-1 accuracy')
    
    model_param_path = '/home/ubuntu/Sketch-Recommendation/model_checkpoints/efficientnet_v2_l-59c71312.pth'
    validate(model_param_path, effnet_v2_logger)