_base_ = [
    '../../_base_/models/fashionformer_r50_fpn.py',
    '../../_base_/default_runtime.py',
]

num_stages = 3
num_proposals = 100
conv_kernel_size = 1
model = dict(
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
    ),
    rpn_head=dict(num_classes=15, ),
    roi_head=dict(
        mask_head=[
            dict(
                type='KernelUpdateHead',
                dataset_type='deepfashion',
                attr_query=True,
                num_classes=15,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=256,
                out_channels=256,
                dropout=0.0,
                mask_thr=0.5,
                conv_kernel_size=conv_kernel_size,
                mask_upsample_stride=2,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(
                    type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0)) for _ in range(num_stages)
        ]),
)

custom_imports = dict(
    imports=[
        'projects.FashionFormer',
        'models.necks.semantic_fpn_wrapper',
        'datasets'
    ],
    allow_failed_imports=False)

dataset_type = 'DeepFashionDataset'
data_root = 'data/DeepFashion/In-shop/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(750, 1101), keep_ratio=True),
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
        img_scale=(750, 1101),
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
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/DeepFashion_segmentation_train.json',
        img_prefix=data_root + 'Img/',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/DeepFashion_segmentation_query.json',
        img_prefix=data_root + 'Img/',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/DeepFashion_segmentation_gallery.json',
        img_prefix=data_root + 'Img/',
        pipeline=test_pipeline,
    ))

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.25)}))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(metric=['segm'])