_base_ = [
    '../__base__/models/fcn_resunet.py',
    '../__base__/datasets/lung.py',
    '../__base__/runtime/default_runtime.py',
    '../__base__/schedules/schedule_10ep_adamw_conv.py'
]

model = dict(
    backbone=dict(
        pretrained='/home/jingfengyao/code/medical/EVA-X/reproduce/pretrained/r50_mgca.pth',
    ),
    decode_head=dict(
        num_classes=2, loss_decode=dict(use_sigmoid=True), out_channels=1),
    auxiliary_head=None,
    test_cfg=dict(mode='whole'))