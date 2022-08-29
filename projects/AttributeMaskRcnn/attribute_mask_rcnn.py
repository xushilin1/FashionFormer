from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import mmcv
import numpy as np
import torch
# from mmdet.core.visualization import imshow_det_bboxes
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mmdet.core.utils import mask2ndarray
EPS = 1e-2

@DETECTORS.register_module()
class AttributeMaskRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(AttributeMaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.ATTR_CLASSES = ('classic', 'polo', 'undershirt', 'henley', 'ringer', 'raglan', 'rugby', 'sailor', 'crop', 'halter', 'camisole', 'tank', 'peasant', 'tube', 
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
    
    def show_result(self,
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
        attrs = np.vstack(attr_result) > 0.5
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
        img = self.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            attrs,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def imshow_det_bboxes(self, 
                    img,
                    bboxes,
                    labels,
                    segms=None,
                    attrs=None,
                    class_names=None,
                    score_thr=0,
                    bbox_color='green',
                    text_color='green',
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=True,
                    wait_time=0,
                    out_file=None):
        assert bboxes.ndim == 2, f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
        assert labels.ndim == 1, f' labels ndim should be 1, but its ndim is {labels.ndim}.'
        assert bboxes.shape[0] == labels.shape[0], 'bboxes.shape[0] and labels.shape[0] should have the same length.'
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
        img = mmcv.imread(img).astype(np.uint8)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
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

        polygons = []
        color = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            bbox_int = bbox.astype(np.int32)
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            # polygons.append(Polygon(np_poly))
            # color.append(bbox_color)
            label_text = ''
            # for idx in attrs[i].nonzero()[0]:
            #     label_text = label_text +  self.ATTR_CLASSES[idx] + '\n'
            # label_text = class_names[label] if class_names is not None else f'class {label}'
            # if len(bbox) > 4:
            #     label_text += f'|{bbox[-1]:.02f}'
            # ax.text(
            #     bbox_int[0],
            #     bbox_int[1],
            #     f'{label_text}',
            #     bbox={
            #         'facecolor': 'black',
            #         'alpha': 0.8,
            #         'pad': 0.7,
            #         'edgecolor': 'none'
            #     },
            #     color=text_color, 
            #     fontsize=font_size,
            #     verticalalignment='top',
            #     horizontalalignment='left')
            if segms is not None:
                color_mask = mask_colors[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

        plt.imshow(img)

        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=thickness)
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

    def color_val_matplotlib(self, color):
        color = mmcv.color_val(color)
        color = [color / 255 for color in color[::-1]]
        return tuple(color)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs