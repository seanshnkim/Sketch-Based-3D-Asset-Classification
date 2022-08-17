import torch

import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier
from torchvision import models, transforms

def test():
    # 1. Create the DataModule
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        batch_size=4,
        transform_kwargs={"image_size": (196, 196), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    )

    # 2. Build the task
    model = ImageClassifier(backbone="resnet18", labels=datamodule.labels)

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
    trainer.finetune(model, datamodule=datamodule, strategy="freeze_unfreeze")

    # 4. Predict what's on a few images! ants or bees?
    datamodule = ImageClassificationData.from_files(
        predict_files=[
            "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
            "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
            "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
        ],
        batch_size=3,
    )
    predictions = trainer.predict(model, datamodule=datamodule, output="labels")
    print(predictions)

    # 5. Save the model!
    trainer.save_checkpoint("image_classification_model.pt")
    
    
    
    
def test2():
    from torchvision import transforms as T

    from typing import Callable, Tuple, Union
    import flash
    from flash.image import ImageClassificationData, ImageClassifier
    from flash.core.data.io.input_transform import InputTransform
    from dataclasses import dataclass


    @dataclass
    class ImageClassificationInputTransform(InputTransform):

        image_size: Tuple[int, int] = (196, 196)
        mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406)
        std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225)

        def input_per_sample_transform(self):
            return T.Compose([T.ToTensor(), T.Resize(self.image_size), T.Normalize(self.mean, self.std)])

        def train_input_per_sample_transform(self):
            return T.Compose(
                [
                    T.ToTensor(),
                    T.Resize(self.image_size),
                    T.Normalize(self.mean, self.std),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(),
                    T.RandomAutocontrast(),
                    T.RandomPerspective(),
                ]
            )

        def target_per_sample_transform(self) -> Callable:
            return torch.as_tensor


    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        train_transform=ImageClassificationInputTransform,
        transform_kwargs=dict(image_size=(128, 128)),
        batch_size=1,
    )

    model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)

    trainer = flash.Trainer(max_epochs=1)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")


def finetune():

    # datamodule = ImageClassificationData.from_folders(
    #     train_folder="data/hymenoptera_data/train/",
    #     val_folder="data/hymenoptera_data/val/",
    #     batch_size=4,
    #     transform_kwargs={"image_size": (196, 196), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    # )
    
    from flash import Trainer
    from flash.image import ImageClassifier, ImageClassificationData
    import os, sys
    
    def find_classes(class_fPath):
        classes_dict = {}
        with open(class_fPath, 'r') as f:
            for s in f.readlines():            
                class_id, class_name = s.split(', ')
                classes_dict[int(class_id)] = class_name
        return classes_dict
    
    idx2classes_dict = find_classes('data/imagenet_classes.txt')
    
    def read_dataset(data_fPath, class_dict, type='val'): # val.txt
        image_files, label_ids, label_names = [], [], []
        with open(data_fPath, 'r') as f:
            for s in f.readlines():
                image_path, label_num = s.rstrip('\n').split(' ')
                label_num = int(label_num)
                image_files.append(
                    os.path.join(os.path.dirname(data_fPath), f"ImageNet_data/ImageNet_{type}", image_path))
                label_ids.append(label_num)
                label_names.append(class_dict[label_num])
        return image_files, label_ids, label_names
    
    
    train_files, _, train_labels = read_dataset('data/val.txt', idx2classes_dict)
    
    
    from typing import Callable, Tuple, Union
    import flash
    from flash.image import ImageClassificationData, ImageClassifier
    from flash.core.data.io.input_transform import InputTransform
    from dataclasses import dataclass
    from torchvision import transforms as T


    @dataclass
    class ImageClassificationInputTransform(InputTransform):
        # TODO: change hyperparams
        image_size: Tuple[int, int] = (196, 196)
        mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406)
        std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225)

        def input_per_sample_transform(self):
            return T.Compose([T.ToTensor(), T.Resize(self.image_size), T.Normalize(self.mean, self.std)])

        def train_input_per_sample_transform(self):
            return T.Compose(
                [
                    T.ToTensor(),
                    T.Resize(self.image_size),
                    T.Normalize(self.mean, self.std),
                    T.RandomAutocontrast(),
                    T.RandomPerspective(),
                ]
            )

        def target_per_sample_transform(self) -> Callable:
            return torch.as_tensor
    
    dm = ImageClassificationData.from_folders(
        train_folder="data/ImageNet_data/ImageNet_train",
        transform_kwargs=dict(image_size=(480, 480), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        batch_size=4
    )
    
    # dm = ImageClassificationData.from_files(
    #     train_files=train_files,
    #     train_targets=train_labels,
    #     # train_transform=ImageClassificationInputTransform,
        
    #     # transform= transforms.Compose([transforms.ToTensor(), 
    #     #                 models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()]))
        
    #     transform_kwargs=dict(image_size=(480, 480), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #     batch_size=4
    # )
    
    model = ImageClassifier(
        backbone="tf_efficientnetv2_l_in21ft1k", 
        # num_classes=dm.num_classes, 
        num_classes=3,
        pretrained=True,
        # labels=['n01440764', 'n01514668', 'n01558993'],
        )

    trainer = flash.Trainer(max_epochs=3, gpus=1)
    trainer.finetune(model, datamodule=dm, strategy=('freeze_unfreeze', 3))

    # dm_predict = ImageClassificationData.from_files(
    #     predict_files=[
    #         "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
    #         "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
    #         "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
    #     ],
    #     batch_size=3,
    # )
    # predictions = trainer.predict(model, datamodule=datamodule, output="labels")
    # print(predictions)

    # 5. Save the model!
    trainer.save_checkpoint("image_classification_model.pt")
    
    

if __name__ == "__main__":
    finetune()
    
    
    
    
    
    