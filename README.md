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
You can replace the batch normalizations with cross-sensor normalizations as follows:
```python
from module.csn import replace_bn_with_csn
from module.semantic_fpn import SemanticFPN
# Semantic-FPN as an example
model = SemanticFPN(dict())
# Replace the BNs with CSNs
model = replace_bn_with_csn(model)
```



### Prepare Cross-Sensor Dataset

```bash
ln -s </path/to/cross_sensor> ./cross_sensor
```

### Eval LoveCS Model
From Airborne to Spaceborne
```bash 
bash ./scripts/eval_lovecs.sh
```


