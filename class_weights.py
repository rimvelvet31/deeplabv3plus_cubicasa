import numpy as np
import torch
from tqdm import tqdm

from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import Compose, ResizePaddedTorch, DictToTensor

def calculate_class_weights(dataset, num_classes, label_index, use_log=True, device='cuda'):
    """
    Calculate class weights based on the dataset.

    Args:
        dataset: The dataset object containing images and labels.
        num_classes: The number of classes in the segmentation task.
        label_index: Index of the specific label tensor (e.g., room or icon segmentation).
        use_log: Whether to use logarithmic scaling for weights.
        device: The device to store the class weights tensor.

    Returns:
        A torch tensor of class weights.
    """

    class_counts = np.zeros(num_classes, dtype=np.float32)
    
    # Iterate over the dataset and count class occurrences
    for sample in tqdm(dataset, desc='Computing class weights', unit='samples', total=len(dataset)):
        labels = sample['label'][label_index].flatten().cpu().numpy()
        for i in range(num_classes):
            class_counts[i] += np.sum(labels == i)
    
    # Compute weights based on frequency (inverse or log-scaled)
    if use_log:
        total_pixels = np.sum(class_counts)
        class_weights = np.log(total_pixels / (class_counts + 1e-6))
    else:
        class_weights = 1.0 / (class_counts + 1e-6)

    # Normalize weights (optional)
    class_weights /= np.sum(class_weights)
    
    return torch.tensor(class_weights, dtype=torch.float32).to(device)


if __name__ == "__main__":
    img_size = 256
    print("Image size:", img_size)

    aug = Compose([
        ResizePaddedTorch((0, 0), data_format='dict', size=(img_size, img_size)),
        DictToTensor()
    ])
    
    dataset = FloorplanSVG('data/cubicasa5k/', 'train.txt', format='lmdb', augmentations=aug)
    
    room_weights = calculate_class_weights(dataset, num_classes=12, label_index=21, use_log=False)
    icon_weights = calculate_class_weights(dataset, num_classes=11, label_index=22, use_log=False)

    print("Room class weights:", room_weights)
    print("Icon class weights:", icon_weights)
