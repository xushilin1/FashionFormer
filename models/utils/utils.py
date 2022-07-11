import torch
import io

from panopticapi.utils import id2rgb
from PIL import Image


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def part_inst2inst_mask(gt_part):
    gt_part_seg = torch.zeros_like(gt_part[0])
    for i in range(gt_part.shape[0]):
        gt_part_seg = torch.where(gt_part[i] != 0, gt_part[i], gt_part_seg)
    classes = gt_part.unique()
    ins_masks = []
    ins_labels = []
    for i in classes:
        ins_labels.append(i)
        ins_masks.append(gt_part_seg == i)
    ins_labels = torch.stack(ins_labels)
    ins_masks = torch.stack(ins_masks)
    return ins_labels.long(), ins_masks.float()


def sem2ins_masks(gt_sem_seg, label_shift=0, thing_label_in_seg=[]):
    """Convert semantic segmentation mask to binary masks
    Args:
        gt_sem_seg (torch.Tensor): Semantic masks to be converted.
            [0, num_thing_classes-1] is the classes of things,
            [num_thing_classes:] is the classes of stuff.
        num_thing_classes (int, optional): Number of thing classes.
            Defaults to 80.
    Returns:
        tuple[torch.Tensor]: (mask_labels, bin_masks).
            Mask labels and binary masks of stuff classes.
    """
    classes = torch.unique(gt_sem_seg)
    # classes ranges from 0 - N-1, where the class IDs in
    # [0, num_thing_classes - 1] are IDs of thing classes
    masks = []
    labels = []

    for i in classes:
        # skip ignore class 255 and "thing classes" in semantic seg
        if i == 255 or i in thing_label_in_seg:
            continue
        labels.append(i)
        masks.append(gt_sem_seg == i)

    if len(labels) > 0:
        labels = torch.stack(labels) + label_shift
        masks = torch.cat(masks)
    else:
        labels = gt_sem_seg.new_zeros(size=[0])
        masks = gt_sem_seg.new_zeros(size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return labels.long(), masks.float()


def encode_panoptic(panoptic_results):
    panoptic_img, segments_info = panoptic_results
    with io.BytesIO() as out:
        Image.fromarray(id2rgb(panoptic_img)).save(out, format='PNG')
        return out.getvalue(), segments_info



def preprocess_panoptic_gt(gt_labels, gt_masks, gt_semantic_seg, num_things,
                           num_stuff):
    """Preprocess the ground truth for a image.

    Args:
        gt_labels (Tensor): Ground truth labels of each bbox,
            with shape (num_gts, ).
        gt_masks (BitmapMasks): Ground truth masks of each instances
            of a image, shape (num_gts, h, w).
        gt_semantic_seg (Tensor): Ground truth of semantic
            segmentation with the shape (1, h, w).
            [0, num_thing_class - 1] means things,
            [num_thing_class, num_class-1] means stuff,
            255 means VOID.
        target_shape (tuple[int]): Shape of output mask_preds.
            Resize the masks to shape of mask_preds.

    Returns:
        tuple: a tuple containing the following targets.

            - labels (Tensor): Ground truth class indices for a
                image, with shape (n, ), n is the sum of number
                of stuff type and number of instance in a image.
            - masks (Tensor): Ground truth mask for a image, with
                shape (n, h, w).
    """
    num_classes = num_things + num_stuff
    things_labels = gt_labels
    gt_semantic_seg = gt_semantic_seg.squeeze(0)

    things_masks = gt_masks.pad(gt_semantic_seg.shape[-2:], pad_val=0)\
        .to_tensor(dtype=torch.bool, device=gt_labels.device)

    semantic_labels = torch.unique(
        gt_semantic_seg,
        sorted=False,
        return_inverse=False,
        return_counts=False)
    stuff_masks_list = []
    stuff_labels_list = []
    for label in semantic_labels:
        if label < num_things or label >= num_classes:
            continue
        stuff_mask = gt_semantic_seg == label
        stuff_masks_list.append(stuff_mask)
        stuff_labels_list.append(label)

    if len(stuff_masks_list) > 0:
        stuff_masks = torch.stack(stuff_masks_list, dim=0)
        stuff_labels = torch.stack(stuff_labels_list, dim=0)
        labels = torch.cat([things_labels, stuff_labels], dim=0)
        masks = torch.cat([things_masks, stuff_masks], dim=0)
    else:
        labels = things_labels
        masks = things_masks

    masks = masks.long()
    return labels, masks
