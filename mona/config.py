config = {
    "height": 32,
    "train_width": 384,
    "batch_size": 256, # 550 on 4GB gpu memo
    "epoch": 1000000,
    "print_per": 10,
    "save_per": 200,

    "train_size": 2560,
    "validate_size": 25600,

    "pretrain": True,
    "pretrain_name": "model_training.pt",

    # Set according to your CPU
    "dataloader_workers": 3,
    # Generate data online for train/val
    "online_train": True,
    "online_val": True,

    # data distribution: genshin / genshin+word 
    "data_only_genshin": True,
    # 生成混合数据时标注数据比例
    "data_genshin_ratio": 0.15,

    # save acc 9999
    "save_acc9999": True,

    # backbone freeze
    # 初始训练时冻结backbone，训练一段时间后解冻
    "freeze_backbone": False,
    "unfreeze_backbone_epoch": 200,
}
