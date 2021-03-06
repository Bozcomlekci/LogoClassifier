# dataset settings
dataset_type = 'Logolist'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/logo_data/train',
        pipeline=train_pipeline,
        ann_file='data/logo_data/meta/train.txt'),
    val=dict(
        type=dataset_type,
        data_prefix='data/logo_data/val',
        ann_file='data/logo_data/meta/val.txt',
        pipeline=test_pipeline),
    test = dict(
        type=dataset_type,
        data_prefix='data/logo_data/val',
        ann_file='data/logo_data/meta/val.txt',
        pipeline=test_pipeline)
    )
evaluation = dict(interval=1, metric='accuracy')
