import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torchmetrics import Accuracy, JaccardIndex

from floortrans.loaders.augmentations import Compose, ResizePaddedTorch, DictToTensor
from floortrans.loaders.svg_loader import FloorplanSVG

from model.deeplabv3plus import DeepLabV3Plus

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path, use_attention=False, device="cuda"):
    model = DeepLabV3Plus(backbone='mobilenetv2', attention=use_attention)

    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    model.to(device)

    return model


def load_dataset(data_folder, image_size=(256, 256)):
    preprocessing = Compose([
        ResizePaddedTorch((0, 0), data_format='dict', size=image_size),
        DictToTensor()
    ])

    txt_sets = ['train.txt', 'val.txt', 'test.txt']
    format = 'lmdb'
    datasets = []

    for txt_set in txt_sets:
        dataset = FloorplanSVG(
            data_folder=data_folder, 
            data_file=txt_set,
            format=format,
            augmentations=preprocessing
        )
        datasets.append(dataset)
    
    combined_dataset = ConcatDataset(datasets)
    return combined_dataset


def load_img_and_labels(dataset, folder_path):
    relative_folder = os.path.relpath(folder_path, 'C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/data/cubicasa5k/')
    relative_folder = relative_folder.replace('\\', '/')
    relative_folder = f'/{relative_folder}/'

    all_folders = np.concatenate([dset.folders for dset in dataset.datasets])
    index = np.where(all_folders == relative_folder)[0][0]
    
    sample = dataset[index]
    img = sample['image']
    labels = sample['label']

    return img, labels


def evaluate(model, img, labels):
    model.eval()

    with torch.no_grad():
        # Add batch dimension
        img = img.unsqueeze(0)
        labels = labels.unsqueeze(0)

        img = img.float().to(DEVICE)
        room_labels = labels[0, 21].long().to(DEVICE)
        icon_labels = labels[0, 22].long().to(DEVICE)

        room_mpa = Accuracy(task='multiclass', num_classes=12, average='macro').to(DEVICE)
        room_miou = JaccardIndex(task='multiclass', num_classes=12, average='macro').to(DEVICE)
        room_fwiou = JaccardIndex(task='multiclass', num_classes=12, average='weighted').to(DEVICE)

        icon_mpa = Accuracy(task='multiclass', num_classes=11, average='macro').to(DEVICE)
        icon_miou = JaccardIndex(task='multiclass', num_classes=11, average='macro').to(DEVICE)
        icon_fwiou = JaccardIndex(task='multiclass', num_classes=11, average='weighted').to(DEVICE)
        
        room_output, icon_output, heatmap_output = model(img)

        room_preds = room_output.argmax(dim=1).squeeze(0)
        icon_preds = icon_output.argmax(dim=1).squeeze(0)

        room_mpa_val = room_mpa(room_preds, room_labels)
        room_miou_val = room_miou(room_preds, room_labels)
        room_fwiou_val = room_fwiou(room_preds, room_labels)

        icon_mpa_val = icon_mpa(icon_preds, icon_labels)
        icon_miou_val = icon_miou(icon_preds, icon_labels)
        icon_fwiou_val = icon_fwiou(icon_preds, icon_labels)

        # print(room_preds.unsqueeze(0).shape)
        # print(icon_preds.unsqueeze(0).shape)
        # print(heatmap_output.squeeze(0).shape)

        combined_tensor = torch.cat([heatmap_output.squeeze(0), room_preds.unsqueeze(0), icon_preds.unsqueeze(0)], dim=0)
        
        return (combined_tensor, [['room_mpa', room_mpa_val], ['room_miou', room_miou_val], ['room_fwiou', room_fwiou_val],
                          ['icon_mpa', icon_mpa_val], ['icon_miou', icon_miou_val], ['icon_fwiou', icon_fwiou_val]])


# if __name__ == '__main__':
    # load_model(r'C:\Users\Red\Documents\GitHub\deeplabv3plus_cubicasa\tool\deeplab\best_checkpoint_mobilenetv2_base.pt')
    # dataset = load_dataset('C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/data/cubicasa5k/')

    # img, labels = load_img_and_labels(dataset, 'C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/data/cubicasa5k/high_quality/90/')
    
    # model = load_model(r'C:\Users\Red\Documents\GitHub\deeplabv3plus_cubicasa\tool\deeplab\best_checkpoint_mobilenetv2_base.pt')

    # evaluate(model, img, labels)

    
    # 'C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/data/cubicasa5k/high_quality/90/'





