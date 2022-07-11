_base_ = [
    '../kfashion_r50_separated_query_base.py'
]

num_stages = 3
num_proposals = 100
conv_kernel_size = 1
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
    ),
    rpn_head=dict(num_classes=46, ),
    roi_head=dict(
        mask_head=[
            dict(
                type='KernelUpdateHead',
                spatial_fusion=False,
                use_mlvl_feat=True,
                attr_query=True,
                num_classes=46,
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


data = dict(
    train=dict(
        times=3,
    ),
)