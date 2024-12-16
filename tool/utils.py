import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torchmetrics import Accuracy, JaccardIndex, MetricCollection
from icecream import ic

from floortrans.loaders.augmentations import Compose, ResizePaddedTorch, DictToTensor
from floortrans.loaders.svg_loader import FloorplanSVG

from model.deeplabv3plus import DeepLabV3Plus


def load_model(model_path, backbone='efficientnet_b2', use_attention=False, device="cuda"):
    model = DeepLabV3Plus(backbone=backbone, attention=use_attention)

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
    dataset_path = os.path.join(os.getcwd(), 'data', 'cubicasa5k')
    
    relative_folder = os.path.relpath(folder_path, dataset_path)
    relative_folder = relative_folder.replace('\\', '/')
    relative_folder = f'/{relative_folder}/'

    all_folders = np.concatenate([dset.folders for dset in dataset.datasets])
    index = np.where(all_folders == relative_folder)[0][0]
    
    sample = dataset[index]
    img = sample['image']
    labels = sample['label']

    return img, labels


def compute_combined_metrics(room_class_metrics, icon_class_metrics, combined_class_freq):
    combined_acc = torch.cat([room_class_metrics['acc'].compute(), icon_class_metrics['acc'].compute()])
    combined_iou = torch.cat([room_class_metrics['iou'].compute(), icon_class_metrics['iou'].compute()])

    combined_mpa = combined_acc.mean()
    combined_miou = combined_iou.mean()

    # fwiou
    total_pixels = combined_class_freq.sum()
    combined_fwiou = (combined_class_freq / total_pixels * combined_iou).sum()

    return {
        'mpa': combined_mpa.item(),
        'cpa': combined_acc.tolist(),
        'miou': combined_miou.item(),
        'fwiou': combined_fwiou.item()
    }


def evaluate(model, img, labels, device='cuda'):
    model.eval()

    # Classes
    class_labels = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", 
                    "Railing", "Storage", "Garage", "Other rooms", "Empty", "Window", "Door", "Closet", 
                    "Electr. Appl.", "Toilet", "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney"]

    # For computing combined metrics
    room_class_metrics = MetricCollection({
        'acc': Accuracy(task='multiclass', num_classes=12, average=None),
        'iou': JaccardIndex(task='multiclass', num_classes=12, average=None)     
    }).to(device)

    icon_class_metrics = MetricCollection({
        'acc': Accuracy(task='multiclass', num_classes=11, average=None),
        'iou': JaccardIndex(task='multiclass', num_classes=11, average=None)     
    }).to(device)

    # To compute combined fwiou
    combined_class_freq = torch.zeros(23).to(device)

    # Add batch dimension
    img = img.unsqueeze(0)
    labels = labels.unsqueeze(0)

    # Convert to appropriate dtype and move to device
    img = img.float().to(device)
    room_labels = labels[0, 21].long().to(device)
    icon_labels = labels[0, 22].long().to(device)

    with torch.no_grad():
        # Get raw outputs
        room_logits, icon_logits, heatmap_logits = model(img)

        # Get predictions
        room_preds = room_logits.argmax(dim=1).squeeze(0)
        icon_preds = icon_logits.argmax(dim=1).squeeze(0)

        # Update metrics
        room_class_metrics(room_preds, room_labels)
        icon_class_metrics(icon_preds, icon_labels)

        # Update combined class frequency
        combined_class_freq[:12] += torch.bincount(room_labels.flatten(), minlength=12)
        combined_class_freq[12:] += torch.bincount(icon_labels.flatten(), minlength=11)
    
    # Compute combined metrics
    metric_scores = compute_combined_metrics(room_class_metrics, icon_class_metrics, combined_class_freq)
    mpa = metric_scores['mpa']
    miou = metric_scores['miou']
    fwiou = metric_scores['fwiou']
    cpa = [val for val in metric_scores['cpa']]
    
    # Combined model predictions
    combined_preds_tensor = torch.cat([room_preds.unsqueeze(0), icon_preds.unsqueeze(0)], dim=0)

    # Combined labels
    combined_labels_tensor = torch.cat([room_labels.unsqueeze(0), icon_labels.unsqueeze(0)], dim=0)

    # Map class labels to class pixel accuracy
    metrics_list = [[class_labels[i], round(cpa[i], 4)] for i in range(len(class_labels))]
    metrics_list.extend([
        ['Mean Pixel Accuracy', round(mpa, 4)],
        ['Mean Intersection over Union', round(miou, 4)],
        ['Frequency-Weighted Intersection over Union', round(fwiou, 4)]
    ])
    # ic(metrics_list)

    return (combined_preds_tensor, combined_labels_tensor, metrics_list)


# if __name__ == '__main__':
    # load_model(r'C:\Users\Red\Documents\GitHub\deeplabv3plus_cubicasa\tool\deeplab\best_checkpoint_mobilenetv2_base.pt')
    # dataset = load_dataset('C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/data/cubicasa5k/')

    # img, labels = load_img_and_labels(dataset, 'C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/data/cubicasa5k/high_quality/90/')
    
    # model = load_model(r'C:\Users\Red\Documents\GitHub\deeplabv3plus_cubicasa\tool\deeplab\best_checkpoint_mobilenetv2_base.pt')

    # evaluate(model, img, labels)

    
    # 'C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/data/cubicasa5k/high_quality/90/'





