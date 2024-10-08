解决无法显示中文问题的方法：

```shell
sudo apt-get update  
sudo apt-get install language-pack-zh-hans language-pack-zh-hans-base
```
HF镜像
```shell
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_CACHE=/root/autodl-tmp/data/cache
```



# 调参记录

## baseline
epoch:1 2卡4090
ALL TRAIN LOGO RECALL:  94.5
ALL TRAIN BRAND RECALL:  86.8519342149479
ALL TEST LOGO RECALL:  20.75
none          97.600000
none_brand    17.004049


## baseline1
在baseline的基础上，使用人工再次标注的数据。
ALL TRAIN LOGO RECALL:  94.75
ALL TRAIN BRAND RECALL:  86.64738958403342
ALL TEST LOGO RECALL:  29.25
none          97.400000
none_brand    14.979757

## baseline2
在baseline1基础上，使用3epoch训练
ALL TRAIN LOGO RECALL:  96.5
ALL TRAIN BRAND RECALL:  88.08314719273625
ALL TEST LOGO RECALL:  23.5
none          99.200000
none_brand     2.024291

## baseline_train2
使用更多品牌数量，但是每个品牌的图片数量变少
ALL TRAIN LOGO RECALL:  93.75
ALL TRAIN BRAND RECALL:  85.18313434409325
ALL TEST LOGO RECALL:  37.75
none          99.200000
none_brand     4.453441


# baseline3
使用更多品牌数据。
ALL TRAIN LOGO RECALL:  87.25
ALL TRAIN BRAND RECALL:  86.39527202883366
ALL TEST LOGO RECALL:  78.75
none          92.800000
none_brand    42.105263
none_brand_label    40.48

# baseline3_e3
这里none_brand有提升，但是由于none_brand并没有标签，所以不确定这个指标的可信度。但是
ALL TRAIN LOGO RECALL:  78.75
ALL TRAIN BRAND RECALL:  79.02586283579434
ALL TEST LOGO RECALL:  75.75
none          93.400000
none_brand    50.607287
none_brand_label    32.79



# baseline3_lorar
https://wandb.ai/nioolek/huggingface/runs/l2bunx5x
将lora_r从128降低为64
ALL TRAIN LOGO RECALL:  76.25
ALL TRAIN BRAND RECALL:  77.28866864997002
ALL TEST LOGO RECALL:  78.25
none          90.200000
none_brand    58.299595
none_brand_label    44.53


# baseline3_lorar1
https://wandb.ai/nioolek/huggingface/runs/bpboge5a
将lora_r从128降低为32
ALL TRAIN LOGO RECALL:  77.5
ALL TRAIN BRAND RECALL:  78.28193464323601
ALL TEST LOGO RECALL:  77.0
none          88.800000
none_brand    58.704453
none_brand_label    42.51

# baseline3_lorar2
https://wandb.ai/nioolek/huggingface/runs/mjupplth
将lora_r从128降低为16
ALL TRAIN LOGO RECALL:  78.0
ALL TRAIN BRAND RECALL:  78.79901016031152
ALL TEST LOGO RECALL:  78.0
none          88.800000
none_brand    62.348178
none_brand_label    46.15


# baseline5
使用crop数据
```
ALL TRAIN LOGO RECALL:  76.5
ALL TRAIN BRAND RECALL:  75.21259562355453
ALL TEST LOGO RECALL:  73.75
none          88.400000
none_brand    60.728745
none_brand_label    47.368421
baseline5和baseline3只有5是使用了crop后的数据集这一个特性。测试集从78.75降低到了73.75.但是none_brand从42增加到了60.
难道是训练集有太多提示性的内容？要不标注下测试集
```

# baseline6
添加商品描述数据 r=128, r_alpha=256
ALL TRAIN LOGO RECALL:  74.5
ALL TRAIN BRAND RECALL:  72.92197908636265
ALL TEST LOGO RECALL:  74.25
none          86.400000
none_brand    69.230769
none_brand_label    52.226720647773284

# baseline6_lorar
r减小一半 r=64, r_alpha=128
ALL TRAIN LOGO RECALL:  75.25
ALL TRAIN BRAND RECALL:  73.6261794395356
ALL TEST LOGO RECALL:  73.75
NONE RECALL:  84.8
NONE BRAND RECALL:  67.61133603238866
NONE BRAND LABEL RECALL:  48.178137651821864

# baseline6_lorar1
r增加一倍 r=256, r_alpha=512
ALL TRAIN LOGO RECALL:  73.5     # 降低
ALL TRAIN BRAND RECALL:  71.25857399830004 # 降低
ALL TEST LOGO RECALL:  75.75 # 增加
NONE RECALL:  78.0  # 变坏
NONE BRAND RECALL:  79.75708502024291
NONE BRAND LABEL RECALL:  55.87044534412956

# result/baseline6_lorar2.csv
r=64, r_alpha=64
ALL TRAIN LOGO RECALL:  72.25
ALL TRAIN BRAND RECALL:  70.25908547312658
ALL TEST LOGO RECALL:  75.0
NONE RECALL:  83.6
NONE BRAND RECALL:  70.44534412955466
NONE BRAND LABEL RECALL:  50.607287449392715

# baseline6_lorar1_nodesc
ALL TRAIN LOGO RECALL:  76.25
ALL TRAIN BRAND RECALL:  74.45348791239202
ALL TEST LOGO RECALL:  71.25
NONE RECALL:  86.2
NONE BRAND RECALL:  62.34817813765182
NONE BRAND LABEL RECALL:  46.963562753036435



# result/baseline6_lorar.csv
ALL TRAIN LOGO RECALL:  75.25
ALL TRAIN BRAND RECALL:  73.6261794395356
ALL TEST LOGO RECALL:  73.75
NONE RECALL:  84.8
NONE BRAND RECALL:  67.61133603238866
NONE BRAND LABEL RECALL:  48.178137651821864

# result/baseline6_lorar_nodesc.csv
ALL TRAIN LOGO RECALL:  75.0
ALL TRAIN BRAND RECALL:  74.50538901737531
ALL TEST LOGO RECALL:  72.0
NONE RECALL:  86.2
NONE BRAND RECALL:  62.75303643724697
NONE BRAND LABEL RECALL:  46.558704453441294

# result/baseline7.csv
基于baseline6，lora只调q和k
ALL TRAIN LOGO RECALL:  69.5
ALL TRAIN BRAND RECALL:  67.10208329215179
ALL TEST LOGO RECALL:  80.0
NONE RECALL:  82.0
NONE BRAND RECALL:  76.11336032388664
NONE BRAND LABEL RECALL:  57.08502024291497

# result/baseline6_e3.csv
ALL TRAIN LOGO RECALL:  78.25
ALL TRAIN BRAND RECALL:  78.26803175262079
ALL TEST LOGO RECALL:  70.25
NONE RECALL:  90.0
NONE BRAND RECALL:  51.012145748987855
NONE BRAND LABEL RECALL:  41.70040485829959


# result/baseline8.csv
ALL TRAIN LOGO RECALL:  74.5
ALL TRAIN BRAND RECALL:  73.17327301060179
ALL TEST LOGO RECALL:  73.75
NONE RECALL:  85.39999999999999
NONE BRAND RECALL:  68.0161943319838
NONE BRAND LABEL RECALL:  51.821862348178136

# llavanext
NONE RECALL:  87.47
NONE BRAND RECALL:  63.56275303643725
NONE BRAND LABEL RECALL:  57.08502024291497

# llavanext1
NONE RECALL:  89.2
NONE BRAND RECALL:  57.08502024291497
NONE BRAND LABEL RECALL:  50.607287449392715

# llavanext2
NONE RECALL:  73.6
NONE BRAND RECALL:  70.44534412955466
NONE BRAND LABEL RECALL:  67.20647773279353

# llavanext3
NONE RECALL:  62.2
NONE BRAND RECALL:  84.21052631578947
NONE BRAND LABEL RECALL:  75.30364372469636

# llavanext4
NONE RECALL:  63.2
NONE BRAND RECALL:  84.21052631578947
NONE BRAND LABEL RECALL:  77.7327935222672


# llavanext5
NONE RECALL:  52.6
NONE BRAND RECALL:  89.47368421052632
NONE BRAND LABEL RECALL:  81.37651821862349

7

8
https://wandb.ai/nioolek/huggingface/runs/ovzrxitl