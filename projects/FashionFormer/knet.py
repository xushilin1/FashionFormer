import torch
import torch.nn.functional as F
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector

from models.utils import sem2ins_masks

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
import mmcv
import numpy as np
import matplotlib.pyplot as plt

import pycocotools.mask as mask_util
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

EPS = 1e-2

@DETECTORS.register_module()
class FashionFormer(TwoStageDetector):

    def __init__(self,
                 *args,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 thing_label_in_seg=list(range(0, 80)),
                 **kwargs):
        super(FashionFormer, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'KNet does not support external proposals'
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.attr_classes = ('classic', 'polo', 'undershirt', 'henley', 'ringer', 'raglan', 'rugby', 'sailor', 'crop', 'halter', 'camisole', 'tank', 'peasant', 'tube', 
        'tunic', 'smock', 'hoodie', 'blazer', 'pea', 'puffer', 'biker', 'trucker', 'bomber', 'anorak', 'safari', 'mao', 'nehru', 'norfolk', 'classic military', 'track', 
        'windbreaker', 'chanel', 'bolero', 'tuxedo', 'varsity', 'crop', 'jeans', 'sweatpants', 'leggings', 'hip-huggers', 'cargo', 'culottes', 'capri', 'harem', 'sailor', 
        'jodhpur', 'peg', 'camo', 'track', 'crop', 'short', 'booty', 'bermuda', 'cargo', 'trunks', 'boardshorts', 'skort', 'roll-up', 'tie-up', 'culotte', 'lounge', 'bloomers', 
        'tutu', 'kilt', 'wrap', 'skater', 'cargo', 'hobble', 'sheath', 'ball gown', 'gypsy', 'rah-rah', 'prairie', 'flamenco', 'accordion', 'sarong', 'tulip', 'dirndl', 'godet', 
        'blanket', 'parka', 'trench', 'pea', 'shearling', 'teddy bear', 'puffer', 'duster', 'raincoat', 'kimono', 'robe', 'dress (coat )', 'duffle', 'wrap', 'military', 'swing', 
        'halter', 'wrap', 'chemise', 'slip', 'cheongsams', 'jumper', 'shift', 'sheath', 'shirt', 'sundress', 'kaftan', 'bodycon', 'nightgown', 'gown', 'sweater', 'tea', 'blouson', 
        'tunic', 'skater', 'asymmetrical', 'symmetrical', 'peplum', 'circle', 'flare', 'fit and flare', 'trumpet', 'mermaid', 'balloon', 'bell', 'bell bottom', 'bootcut', 'peg', 
        'pencil', 'straight', 'a-line', 'tent', 'baggy', 'wide leg', 'high low', 'curved', 'tight', 'regular', 'loose', 'oversized', 'empire waistline', 'dropped waistline', 
        'high waist', 'normal waist', 'low waist', 'basque', 'no waistline', 'above-the-hip', 'hip', 'micro', 'mini', 'above-the-knee', 'knee', 'below the knee', 'midi', 'maxi', 
        'floor', 'sleeveless', 'short', 'elbow-length', 'three quarter', 'wrist-length', 'asymmetric', 'regular', 'shirt', 'polo', 'chelsea', 'banded', 'mandarin', 'peter pan', 'bow',
        'stand-away', 'jabot', 'sailor', 'oversized', 'notched', 'peak', 'shawl', 'napoleon', 'oversized', 'collarless', 'asymmetric', 'crew', 'round', 'v-neck', 'surplice', 'oval',
        'u-neck', 'sweetheart', 'queen anne', 'boat', 'scoop', 'square', 'plunging', 'keyhole', 'halter', 'crossover', 'choker', 'high', 'turtle', 'cowl', 'straight across', 'illusion', 
        'off-the-shoulder', 'one shoulder', 'set-in sleeve', 'dropped-shoulder sleeve', 'ragla', 'cap', 'tulip', 'puff', 'bell', 'circular flounce', 'poet', 'dolma, batwing', 'bishop', 
        'leg of mutto', 'kimono', 'cargo', 'patch', 'welt', 'kangaroo', 'seam', 'slash', 'curved', 'flap', 'single breasted', 'double breasted', 'lace up', 'wrapping', 'zip-up', 
        'fly', 'chained', 'buckled', 'toggled', 'no opening', 'plastic', 'rubber', 'metal', 'feather', 'gem', 'bone', 'ivory', 'fur', 'suede', 'shearling', 'crocodile', 'snakeskin', 
        'wood', 'non-textile material', 'burnout', 'distressed', 'washed', 'embossed', 'frayed', 'printed', 'ruched', 'quilted', 'pleat', 'gathering', 'smocking', 'tiered', 'cutout', 
        'slit', 'perforated', 'lining', 'applique', 'bead', 'rivet', 'sequin', 'no special manufacturing technique', 'plain', 'abstract', 'cartoon', 'letters, numbers', 'camouflage',
        'check', 'dot', 'fair isle', 'floral', 'geometric', 'paisley', 'stripe', 'houndstooth', 'herringbone', 'chevron', 'argyle', 'leopard', 'snakeskin', 'cheetah', 'peacock', 
        'zebra', 'giraffe', 'toile de jouy', 'plant')
    

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_semantic_seg=None,
                      gt_attributes=None,
                      **kwargs):
        """Forward function of SparseR-CNN in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        super(TwoStageDetector, self).forward_train(img, img_metas)

        assert proposals is None, 'KNet does not support external proposals'
        # for i in range(len(gt_masks)):
        #     assert gt_labels[i].shape[0] == gt_attributes[i].shape[0], f'{gt_labels[i].shape[0]} {gt_attributes[i].shape[0]} {img_metas[i]}'
        assert gt_masks is not None

        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks_tensor = []
        gt_sem_seg = []
        gt_sem_cls = []
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.mask_assign_stride
        assign_W = pad_W // self.mask_assign_stride

        for i, gt_mask in enumerate(gt_masks):
            mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)

            if gt_semantic_seg is not None:
                # gt_semantic seg is padded by 255 and
                # zero indicating the first class
                gt_semantic_seg[i, :, img_metas[i]['img_shape'][0]:, :] = 255
                gt_semantic_seg[i, :, :, img_metas[i]['img_shape'][1]:] = 255
                sem_labels, sem_seg = sem2ins_masks(gt_semantic_seg[i], thing_label_in_seg=self.thing_label_in_seg)
                if sem_seg.shape[0] == 0:
                    gt_sem_seg.append(mask_tensor.new_zeros((mask_tensor.size(0), assign_H, assign_W)))
                else:
                    gt_sem_seg.append(
                        F.interpolate(
                            sem_seg[None], (assign_H, assign_W),
                            mode='bilinear',
                            align_corners=False)[0])
                gt_sem_cls.append(sem_labels)
            else:
                gt_sem_seg = None
                gt_sem_cls = None

            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                gt_masks_tensor.append(
                    F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0])

        gt_masks = gt_masks_tensor
        x = self.extract_feat(img)
        rpn_results = self.rpn_head.forward_train(x, img_metas, gt_masks,
                                                  gt_labels, gt_sem_seg,
                                                  gt_sem_cls)
        (rpn_losses, proposal_kernels, x_feats, mask_preds, cls_scores, mlvl_feats) = rpn_results

        losses = self.roi_head.forward_train(
            x_feats,
            mlvl_feats,
            proposal_kernels,
            mask_preds,
            cls_scores,
            img_metas,
            gt_masks,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_bboxes=gt_bboxes,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls,
            gt_attributes=gt_attributes,
            imgs_whwh=None)

        losses.update(rpn_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        rpn_results = self.rpn_head.simple_test_rpn(x, img_metas)
        (proposal_kernels, x_feats, mask_preds, cls_scores, seg_preds, mlvl_feats) = rpn_results
        segm_results = self.roi_head.simple_test(
            x_feats,
            mlvl_feats,
            proposal_kernels,
            mask_preds,
            cls_scores,
            img_metas,
            imgs_whwh=None,
            rescale=rescale)
        return segm_results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        rpn_results = self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds, mlvl_feats) = rpn_results
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x_feats, mlvl_feats, mask_preds, proposal_feats, dummy_img_metas)
        return roi_outs

    def _show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result, attr_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        attrs = np.vstack([attr for attr in attr_result if len(attr) > 0]) > 0.5
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = self.imshow_det_bboxes(img, bboxes, labels, segms, attrs, class_names=self.CLASSES, 
                            score_thr=score_thr, bbox_color=bbox_color, text_color=text_color,
                            mask_color=mask_color, thickness=thickness, font_size=font_size, 
                            win_name=win_name, show=show, wait_time=wait_time,out_file=out_file)

        if not (show or out_file):
            return img

    def color_val_matplotlib(self, color):
        color = mmcv.color_val(color)
        color = [color / 255 for color in color[::-1]]
        return tuple(color)

    def imshow_det_bboxes(self, img, bboxes, labels, segms=None, attrs=None, class_names=None, score_thr=0,bbox_color='green', text_color='green',
                        mask_color=None, thickness=2, font_size=13, win_name='', show=True, wait_time=0, out_file=None):
       
        assert bboxes.ndim == 2, ' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
        assert labels.ndim == 1, ' labels ndim should be 1, but its ndim is {labels.ndim}.'
        assert bboxes.shape[0] == labels.shape[0], 'bboxes.shape[0] and labels.shape[0] should have the same length.'
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, ' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
        img = mmcv.imread(img).astype(np.uint8)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            attrs = attrs[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]
        mask_colors = [[[100,  78, 126]],[[136,  37,  47]],[[ 85,  36, 238]],[[193, 213,  73]], [[172, 188,   2]], 
            [[ 63, 196, 237]], [[153, 191,  17]], [[158,  45, 198]], [[182,  50, 243]], [[ 91, 156, 104]], [[140,  37, 146]], 
            [[205,   2, 150]],  [[ 97, 155, 243]],[[ 44, 137,  32]], [[15, 37, 24]], [[111,  33,   6]], [[ 88,  65, 192]], 
            [[245,  14, 230]], [[ 62, 227, 253]], [[154,  23, 197]], [[ 28, 188, 151]], [[  9,  89, 226]], [[ 57, 240, 104]], 
            [[155,  75, 165]], [[138,  57, 162]], [[ 28, 177,  46]], [[102, 177, 173]], [[  0, 110, 167]],[[  2,  96, 220]], 
            [[217,  50, 200]], [[ 26, 172, 208]], [[238, 142,  86]],  [[ 54, 196, 241]], [[120, 145,  79]],[[ 73,  67, 114]], 
            [[ 20, 171,  36]], [[ 52, 105, 116]], [[ 35, 180,  90]], [[182,  48,  97]], [[ 44,  14, 127]], [[ 79,  90, 198]], 
            [[224, 117,  79]], [[ 22, 126,  35]], [[ 27, 203,  96]], [[52,  0, 27]],   [[202, 228,  99]], [[130, 114,  41]], 
            [[ 87, 135, 233]], [[131, 246,  47]],  [[241, 149, 237]]]
        mask_colors = [np.array(c) for c in mask_colors]
        
        bbox_color = self.color_val_matplotlib(bbox_color)
        text_color = self.color_val_matplotlib(text_color)

        img = mmcv.bgr2rgb(img)
        width, height = img.shape[1], img.shape[0]
        img = np.ascontiguousarray(img)

        fig = plt.figure(win_name, frameon=False)
        plt.title(win_name)
        canvas = fig.canvas
        dpi = fig.get_dpi()
        # add a small EPS to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

        # remove white edges by set subplot margin
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis('off')
        from chainercv.utils import mask_to_bbox
        bboxes = mask_to_bbox(segms)
        polygons = []
        color = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            bbox_int = bbox.astype(np.int32)
            color.append(bbox_color)
            label_text = ''
            for idx in attrs[i].nonzero()[0]:
                label_text = label_text +  self.attr_classes[idx] + '\n'
            # label_text = class_names[label] if class_names is not None else f'class {label}'
            # if len(bbox) > 4:
            #     label_text += f'|{bbox[-1]:.02f}'
            ax.text(
                bbox_int[1],
                bbox_int[0],
                f'{label_text}',
                bbox={
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color=text_color, 
                fontsize=font_size,
                verticalalignment='top',
                horizontalalignment='left')
            if segms is not None:
                color_mask = mask_colors[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

        plt.imshow(img)

        p = PatchCollection(
            polygons, facecolor='none', edgecolors=color, linewidths=thickness)
        ax.add_collection(p)

        stream, _ = canvas.print_to_buffer()
        buffer = np.frombuffer(stream, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        img = rgb.astype('uint8')
        img = mmcv.rgb2bgr(img)

        if show:
            # We do not use cv2 for display because in some cases, opencv will
            # conflict with Qt, it will output a warning: Current thread
            # is not the object's thread. You can refer to
            # https://github.com/opencv/opencv-python/issues/46 for details
            if wait_time == 0:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        plt.close()

        return img