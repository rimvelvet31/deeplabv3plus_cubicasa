import torch
from torchmetrics import MetricCollection, Accuracy, JaccardIndex

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Combined metrics
class_room_metrics = MetricCollection({
    'acc': Accuracy(task='multiclass', num_classes=12, average=None),
    'iou': JaccardIndex(task='multiclass', num_classes=12, average=None)     
}).to(device)

class_icon_metrics = MetricCollection({
    'acc': Accuracy(task='multiclass', num_classes=11, average=None),
    'iou': JaccardIndex(task='multiclass', num_classes=11, average=None)     
}).to(device)

def compute_combined_metrics(class_room_metrics, class_icon_metrics):
    combined_acc = torch.cat([class_room_metrics['acc'].compute(), class_icon_metrics['acc'].compute()])
    combined_iou = torch.cat([class_room_metrics['iou'].compute(), class_icon_metrics['iou'].compute()])

    combined_mpa = combined_acc.mean()
    combined_miou = combined_iou.mean()

    # fwIOU
    all_targets = torch.cat([class_room_metrics['iou']._target, class_icon_metrics['iou']._target])

    _, counts = torch.unique(all_targets, return_counts=True)
    class_freqs = counts.float() / len(all_targets)

    combined_fwiou = (combined_iou * class_freqs).sum() / class_freqs.sum()
    combined_fwiou = combined_fwiou.item()

    return { 'mpa': combined_mpa, 'miou': combined_miou, 'fwiou': combined_fwiou }