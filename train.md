# 配置环境

## pytorch 基础环境 + clone仓库

略，应该都会

## pickle 数据

标注完成的数据格式如下：

```
├── dumps
│   ├── x.json
│   └── y.json
├── dumps4.0_pph
│   ├── x.json
│   └── y.json
├── dumps4.0_syfs
│   ├── x.json
│   └── y.json
├── dumps4.0_tx
│   ├── x.json
│   └── y.json
├── dumpxxxx, 与上面一样
├── xx.json
└── yy.json
```

目前根路径是硬编码在`yap/`下，因为是yap dump的数据，标注脚本也在yap而不是本仓库下（以后有空再解耦吧）。

为了加速，需要先pickle dump一个数据集以供训练时数据生成加速。
使用`make_xy_pickle.py`，生成的`.pkl`路径目前是硬编码在代码中的。


## 训练

直接`py main.py train`, 所有训练相关的参数都在`config.py`里，与yas一致。


但是多了数据分布和生成混合数据时的标注数据比例两个设置。
前者是引入了针对单字的数据，后者是控制生成数据与标注数据的比例。


在预训练权重加载中，使用了`load can load`策略，即当总字数量变动时，网络结构会发生改变，会尽可能加载已有的权重，但是如果不匹配，会重新初始化。这一策略有效提升了训练效率。


## “反哺”策略

使用训练好的模型进行预测，将预测结果与标注结果进行比较，将**标注错误**的数据修改标注，再次训练。见`clean_up.py`