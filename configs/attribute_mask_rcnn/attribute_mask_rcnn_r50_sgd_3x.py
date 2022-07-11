_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/fashionpedia.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py',
]
model = dict(
    type='AttributeMaskRCNN',
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

optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[9, 11])

custom_imports = dict(
    imports=[
        'datasets',
        'projects.AttributeMaskRcnn'
    ],
    allow_failed_imports=False)

evaluation = dict(interval=1, metric=['bbox', 'segm'])
