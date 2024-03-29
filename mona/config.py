config = {
    "device": "cuda",

    "height": 32,
    "train_width": 384,
    "batch_size": 4096, # 550 on 4GB gpu memo
    "epoch": 1000000,
    "print_per": 10,
    "save_per": 100,

    "train_size": 4096000,
    "validate_size": 20000,

    "pretrain": True,
    "pretrain_name": "model_best.pt",
    'lr': 1e-3,

    # Set according to your CPU
    "dataloader_workers": 1,
    # Generate data online for train/val
    "online_train": True,
    "online_val": True,

    # data distribution: genshin pickup / genshin+common chinese&punctuation
    # "data_only_genshin": False,
    "data_only_genshin": True,
    # 生成混合数据时pickup的比例
    "pickup_ratio": 0.9,
    # pickup 中 genshin 真实数据的比例
    "data_genshin_ratio": 0.25,
    
    # save acc 9999
    "save_acc9999": True,

    # backbone freeze
    # 初始训练时冻结backbone，训练一段时间后解冻
    "freeze_backbone": True,
    "unfreeze_backbone_epoch": 1000,
}
