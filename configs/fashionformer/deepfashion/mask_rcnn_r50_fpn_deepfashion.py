_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/deepfashion.py',
    '../../_base_/default_runtime.py'
]


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=15), 
        mask_head=dict(num_classes=15)
    )
)

# runtime settings
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)