import itertools
import logging

import warnings
from collections import OrderedDict

import mmcv
import numpy as np

from mmcv.utils import print_log
from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS

from terminaltables import AsciiTable

from .fp_eval import FPEval, Fashionpedia


@DATASETS.register_module()
class FashionpediaDataset(CocoDataset):
    CLASSES = ('shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket',
               'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat',
               'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer',
               'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood',
               'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead',
               'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel')

    def load_annotations(self, ann_file):
        """Load annotation from COCO Panoptic style annotation file.
        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = Fashionpedia(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.categories = self.coco.cats

        self.attr_ids = self.coco.getAttIds()
        self.attr2label = {attr_id: i for i, attr_id in enumerate(sorted(self.attr_ids))}
        self.attributes = self.coco.attrs
        self.continuous2attid = {v: k for k, v in self.attr2label.items()}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            # info['segm_file'] = info['filename'].replace('jpg', 'png')
            data_infos.append(info)
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_attributes = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                one_hot = [0 for _ in range(len(self.attributes))]
                for x in ann.get('attribute_ids', []):
                    one_hot[self.attr2label[x]] = 1
                gt_attributes.append(one_hot)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_attributes = np.array(gt_attributes, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            attributes=gt_attributes,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            iou_type = metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = FPEval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 'AP',
                'mAP_50': 'AP_IOU50',
                'mAP_75': 'AP_IOU75',
                'mAP_s': 'APs',
                'mAP_m': 'APm',
                'mAP_l': 'APl',
                'AR@100': 'AR@100',
                'AR@300': 'AR@300',
                'AR@1000': 'AR@1000',
                'AR_s@1000': 'ARs@100',
                'AR_m@1000': 'ARm@100',
                'AR_l@1000': 'ARl@100',
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # cocoEval.summarize_class()
            # cocoEval.print()
            # cocoEval.print(False)
            # cocoEval._per_supercls_summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append((f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[results_flatten[i::num_columns] for i in range(num_columns)])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(f'{cocoEval.resultDet[coco_metric_names[metric_item]]:.3f}')
                eval_results[key] = val
            eval_results[f'{metric}_mAP_iou+f1'] = cocoEval.results['AP']
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg, attr = results[idx]
            for label in range(len(det)):

                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    attr_idx = [self.continuous2attid[i] for i, a in enumerate(attr[label][i]) if a > 0.5]
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    data['attribute_ids'] = attr_idx
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    attr_idx = [self.continuous2attid[i] for i, a in enumerate(attr[label][i]) if a > 0.5]
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    data['attribute_ids'] = attr_idx
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results
