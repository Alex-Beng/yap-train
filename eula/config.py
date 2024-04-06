config = {
    "device": "cuda",

    "height": 384,
    "width": 64,
    "num_classes": 2, # 0: background, 1: F for now

    "init_epoch": 0,
    "epoch": 1000,
    "batch_size": 32, 
    "print_per": 10,
    "save_per": 20,

    "train_size": 25600,
    "validate_size": 2560,

    "pretrain": True,
    "pretrain_name": "eula/model_best.pt",
    "dataloader_workers": 1,
    "online_train": True,
    "online_val": True,

}