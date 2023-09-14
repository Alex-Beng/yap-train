config = {
    "height": 32,
    "train_width": 384,
    "batch_size": 512, # 550 on 4GB gpu memo
    "epoch": 1000,
    "print_per": 10,
    "save_per": 400,

    "train_size": 200000,
    "validate_size": 30000,

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
    "data_genshin_ratio": 0.3,
}
