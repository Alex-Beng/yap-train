config = {
    "device": "cuda",

    "height": 32,
    "train_width": 384,
    "batch_size": 64, # 550 on 4GB gpu memo
    "epoch": 1000000,
    "print_per": 10,
    "save_per": 100,

    "train_size": 4096000,
    "validate_size": 12800,

    "pretrain": True,
    "pretrain_name": "model_best.pt",
    'lr': 1e-5,

    # Set according to your CPU
    "dataloader_workers": 1,
    # Generate data online for train/val
    "online_train": True,
    "online_val": True,

    # data distribution: genshin pickup / genshin+common chinese&punctuation
    # "data_only_genshin": False,
    "data_only_genshin": True,
    # 生成混合数据时pickup的比例
    "pickup_ratio": 0.75,
    # pickup 中 genshin 真实数据的比例，以及 validate 中错误的 another 数据比例
    "data_genshin_ratios": [0.3, 0.1],
    
    # save acc 9999
    "save_acc9999": True,

    # backbone freeze
    # 初始训练时冻结backbone，训练一段时间后解冻
    "freeze_backbone": False,
    "unfreeze_backbone_epoch": 1000,

    # 倒反天罡，冻结其他层，训练cnn
    "freeze_withouth_backbone": False,
    "unfreeze_withouth_backbone_epoch": 1000,
}

mb_config = {
    "device": "cuda",

    "height": 32,
    "train_width": 384,
    "batch_size": 256, # 550 on 4GB gpu memo
    "epoch": 1000000,
    "print_per": 10,
    "save_per": 400,

    "train_size": 10240, # 由 batch_size * save_per 算了
    "validate_size": 25600,
    # "validate_size": 256,

    "pretrain": True,
    "pretrain_name": "model_best_.pt",
    'lr': 1e-4,

    # Set according to your CPU
    "dataloader_workers": 1,
    # Generate data online for train/val
    "online_train": True,
    "online_val": True,

    # data distribution: genshin pickup / genshin+common chinese&punctuation
    # "data_only_genshin": False,
    "data_only_genshin": True,
    # 生成混合数据时pickup的比例
    "pickup_ratio": 0.75,
    # pickup 中 genshin 真实数据的比例，以及 validate 中错误的 another 数据比例
    "data_genshin_ratios": [0.4, 0.1],
    
    # save acc 9999
    "save_acc9999": False,

    # backbone freeze
    # 初始训练时冻结backbone，训练一段时间后解冻
    "freeze_backbone": False,
    "unfreeze_backbone_epoch": 1000,

    # 倒反天罡，冻结其他层，训练cnn
    "freeze_withouth_backbone": False,
    "unfreeze_withouth_backbone_epoch": 1000,
}

gt_config = {
    "device": "cuda",

    # 因为是正方形，所以只需要一个边长
    "side_len": 224,
    
    "batch_size": 16,
    "epoch": 1000000,
    "print_per": 10,
    "save_per": 800,

    "train_size": 4096000,
    "validate_size": 5120,

    "pretrain": True,
    "pretrain_name": "model_best.pt",
    'lr': 1e-4,

    # Set according to your CPU
    "dataloader_workers": 1,
    # Generate data online for train/val
    "online_train": True,
    "online_val": True,

    # save acc 9999
    "save_acc9999": True,

    # backbone freeze
    # 初始训练时冻结backbone，训练一段时间后解冻
    "freeze_backbone": False,
    "unfreeze_backbone_epoch": 1000,
}

gt_noexp_config = {
    "device": "cuda",

    # 因为是正方形，所以只需要一个边长
    "side_len": 224,
    
    "batch_size": 128,
    "epoch": 1000000,
    "print_per": 10,
    "save_per": 800,

    "train_size": 4096000,
    "validate_size": 5120,

    "pretrain": False,
    "pretrain_name": "model_best.pt",
    'lr': 1e-5,

    # Set according to your CPU
    "dataloader_workers": 1,
    # Generate data online for train/val
    "online_train": True,
    "online_val": True,

    # save acc 9999
    "save_acc9999": True,

    # backbone freeze
    # 初始训练时冻结backbone，训练一段时间后解冻
    "freeze_backbone": True,
    "unfreeze_backbone_epoch": 1000,
}

maxact_config = {
    "device": "cuda",
    "pretrain_name": "model_best_.pt",
    "lr": 1e-2,
    "iters": 100000,
    "show_per": 10,
    "batch_size": 30,
    "init_tensor_method": "datagen",
    "init_str_method": "rand",
}