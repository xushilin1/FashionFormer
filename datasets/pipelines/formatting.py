import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.pipelines.formating import DefaultFormatBundle

@PIPELINES.register_module()
class FashionpediaFormatBundle(DefaultFormatBundle):
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """
        results = super(FashionpediaFormatBundle, self).__call__(results)
        if 'gt_attributes' in results:
            results['gt_attributes'] = DC(to_tensor(results['gt_attributes']))
        return results