import os.path as osp
import numpy as np
from PIL import Image
import mmcv
import os
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet.core import BitmapMasks, PolygonMasks
import collections
from panopticapi.utils import rgb2id
from mmdet.datasets.pipelines.loading import FilterAnnotations


@PIPELINES.register_module()
class AttributeFilterAnnotations(FilterAnnotations):

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            return None
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_attributes')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results


@PIPELINES.register_module()
class LoadFashionPediaAnnotations(LoadAnnotations):
    def __call__(self, results):
        results = super(LoadFashionPediaAnnotations, self).__call__(results)
        results = self._load_fashionpedia(results)
        return results

    def _load_fashionpedia(self, results):
        ann_info = results['ann_info']
        results['gt_attributes'] = ann_info['attributes'].copy()
        results['attributes_fields'] = ['gt_attributes']
        return results


@PIPELINES.register_module(force=True)
class LoadPanopticAnnotations(LoadAnnotations):
    """Load multiple types of panoptic annotations.
    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 with_seg=True,
                 file_client_args=dict(backend='disk')):
        if rgb2id is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super(LoadPanopticAnnotations,
              self).__init__(with_bbox, with_label, with_mask, with_seg, True, file_client_args)

    def _load_masks_and_semantic_segs(self, results):
        """Private function to load mask and semantic segmentation annotations.
        In gt_semantic_seg, the foreground label is from `0` to
        `num_things - 1`, the background label is from `num_things` to
        `num_things + num_stuff - 1`, 255 means the ignored label (`VOID`).
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded mask and semantic segmentation
                annotations. `BitmapMasks` is used for mask annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])

        img_bytes = self.file_client.get(filename)
        pan_png = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for mask_info in results['ann_info']['masks']:
            mask = (pan_png == mask_info['id'])
            gt_seg = np.where(mask, mask_info['category'], gt_seg)

            # The legal thing masks
            if mask_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))

        if self.with_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

        if self.with_seg:
            results['gt_semantic_seg'] = gt_seg
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types panoptic annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask or self.with_seg:
            # The tasks completed by '_load_masks' and '_load_semantic_segs'
            # in LoadAnnotations are merged to one function.
            results = self._load_masks_and_semantic_segs(results)

        return results


