_base_ = [
    '../kfashion_r50_separated_query_base.py'
]

# add the swin path
# pretrained = '/mnt/lustre/lixiangtai/pretrained/swin/swin_base_patch4_window7_224_22k.pth'
pretrained = None

num_stages = 3
num_proposals = 100
conv_kernel_size = 1
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        times=8,
    ),
)