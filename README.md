解决无法显示中文问题的方法：

```shell
sudo apt-get update  
sudo apt-get install language-pack-zh-hans language-pack-zh-hans-base
```
HF镜像
```shell
export HF_ENDPOINT=https://hf-mirrorcom
```



# 调参记录

## baseline
epoch:1 2卡4090
ALL TRAIN LOGO RECALL:  94.5
ALL TRAIN BRAND RECALL:  86.8519342149479
ALL TEST LOGO RECALL:  9.0
none          97.600000
none_brand    17.004049


## baseline1
在baseline的基础上，使用人工再次标注的数据。
ALL TRAIN LOGO RECALL:  94.75
ALL TRAIN BRAND RECALL:  86.64738958403342
ALL TEST LOGO RECALL:  14.0
none          97.400000
none_brand    14.979757

## baseline2
在baseline1基础上，使用3epoch训练
ALL TRAIN LOGO RECALL:  96.5
ALL TRAIN BRAND RECALL:  88.08314719273625
ALL TEST LOGO RECALL:  10.5
none          99.200000
none_brand     2.024291

## baseline_train2
使用更多品牌数量，但是每个品牌的图片数量变少
ALL TRAIN LOGO RECALL:  93.75
ALL TRAIN BRAND RECALL:  85.18313434409325
ALL TEST LOGO RECALL:  14.75