_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/fashionpedia.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py',
]

image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# add pretrained swin model
pretrained = '/mnt/lustre/lixiangtai/pretrained/swin/swin_base_patch4_window7_224_22k.pth'

model = dict(
    type='AttributeMaskRCNN',
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
        type='AttributeHead',
        bbox_head=dict(
            type='AttributeBBoxHead',
            num_classes=46,
            loss_attr=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        ),
        mask_head=dict(
            num_classes=46,
        )
    ),
)
optimizer = dict(type='AdamW',lr=0.0001,weight_decay=0.05,paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.25)}))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])

custom_imports = dict(
    imports=[
        'datasets',
        'projects.AttributeMaskRcnn'
    ],
    allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFashionPediaAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='AttributeRandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
    ),
    dict(type='AttributeFilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FashionpediaFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_attributes']),
]
evaluation = dict(interval=1, metric=['bbox', 'segm'])

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            pipeline=train_pipeline),
    )
)