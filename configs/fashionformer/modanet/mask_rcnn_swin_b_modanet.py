_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
pretrained='pretrained/swin_base_patch4_window7_224_22k.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        bbox_head=dict(num_classes=13,),
        mask_head=dict(num_classes=13,)
    ),
)


dataset_type = 'CocoDataset'
data_root = 'data/ModaNet/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(400, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(400, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        classes=('bag','belt','boots','footwear','outer','dress','sunglasses', 'pants','top','shorts','skirt','headwear','scarf/tie'),
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        classes=('bag','belt','boots','footwear','outer','dress','sunglasses', 'pants','top','shorts','skirt','headwear','scarf/tie'),
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'train/',
        pipeline=test_pipeline),
    test=dict(
        classes=('bag','belt','boots','footwear','outer','dress','sunglasses', 'pants','top','shorts','skirt','headwear','scarf/tie'),
        type=dataset_type,
        ann_file=data_root + 'annotations/modanet2018_instances_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))


optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0)
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=True,
    step=[50],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)
runner = dict(type='EpochBasedRunner', max_epochs=12)