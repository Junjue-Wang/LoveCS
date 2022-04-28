<h2 align="center">Cross-sensor domain adaptation for high-spatial resolution urban land-cover mapping: from airborne to spaceborne imagery</h2>


<h5 align="right">by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>, 
Ailong Ma,
<a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>, 
<a href="http://zhuozheng.top/">Zhuo Zheng</a>, and Liangpei Zhang</h5>


<div align="center">
  <img src="https://github.com/Junjue-Wang/resources/blob/main/LoveCS/framework.png?raw=true">
</div>

---------------------

This is an official implementation LoveCS in our RSE 2022 paper.
### Requirements:
- pytorch >= 1.7.0
- python >=3.6
```bash
pip install --upgrade git+https://gitee.com/zhuozheng/ever_lite.git@v1.4.5
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```

### 1. Flexible Cross-sensor Normalization
Cross-sensor normalization can help you encode the source and target domain statistics 
separately.

1.You can replace the batch normalizations with cross-sensor normalizations as follows:

```python
from module.csn import replace_bn_with_csn
from module.semantic_fpn import SemanticFPN
# Semantic-FPN (https://arxiv.org/pdf/1901.02446.pdf) as an example 
model = SemanticFPN(dict())
# Replace the BNs with CSNs
model = replace_bn_with_csn(model)
```

2.Model Forward
```python
from module.csn import change_csn
model = change_csn(model, source=True)
source_outputs = model(source_images)
model = change_csn(model, source=False)
target_outputs = model(target_images)
```

### 2. LoveCS framework
[LoveCS_train.py](https://github.com/Junjue-Wang/LoveCS/blob/master/LoveCS_train.py) is a training example and
[LoveCS_eval.py](https://github.com/Junjue-Wang/LoveCS/blob/master/LoveCS_eval.py) is an evaluation example.
You can configure your domain adaptation dataset (i.e. [LoveDA](https://github.com/Junjue-Wang/LoveDA)) and use the following scripts for training and evaluation.
```bash
#!/usr/bin/env bash
config_path='st.lovecs.2CZ.lovecs'
python LoveCS_train.py --config_path=${config_path}


config_path='st.lovecs.2CZ.lovecs'
ckpt_path='./log/sfpn.pth'
python LoveCS_eval.py --config_path=${config_path} --ckpt_path=${ckpt_path}
```
![avatar](https://github.com/Junjue-Wang/resources/blob/main/LoveCS/overall_prcocess.png?raw=true)
