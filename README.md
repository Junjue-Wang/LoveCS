<h2 align="center">Cross-sensor domain adaptation for high-spatial resolution urban land-cover mapping: from airborne to spaceborne imagery</h2>


<h5 align="right">by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>, 
Ailong Ma,
<a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>, 
<a href="http://zhuozheng.top/">Zhuo Zheng</a>, and Liangpei Zhang</h5>

[[`Paper`](https://www.researchgate.net/publication/360484883_Cross-sensor_domain_adaptation_for_high_spatial_resolution_urban_land-cover_mapping_From_airborne_to_spaceborne_imagery)],
[[`BibTex`](https://www.sciencedirect.com/sdfe/arp/cite?pii=S0034425722001729&format=text%2Fx-bibtex&withabstract=true)],
[[`Product`](https://pan.baidu.com/s/1YnsMFDOMBGO-oz_PAUkuFQ?pwd=2333)]

<div align="center">
  <img src="https://github.com/Junjue-Wang/resources/blob/main/LoveCS/framework.png?raw=true">
</div>

---------------------

This is an official implementation LoveCS in our RSE 2022 paper.

## Highlights:
1. A Cross-Sensor Land-cOVEr framework (LoveCS) is proposed.
2. LoveCS advances cross-sensor domain adaptation.
3. LoveCS learns divergence between sensors from structure and optimization.
4. The effectiveness of LoveCS was evaluated in three cities of China.
5. High-resolution city-scale mapping can be achieved within 9 hours on one GPU.

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

2.Model Backward
```python
from torch.nn import CrossEntropyLoss
loss_cal = CrossEntropyLoss()
src_loss = loss_cal(source_outputs, src_labels)
tgt_loss = loss_cal(target_outputs, pse_labels)
total_loss = tgt_loss + src_loss
total_loss.backward()
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

## Citation
If you use LoveCS in your research, please cite our RSE 2022 paper.
```text
    @article{WANG2022113058,
    title = {Cross-sensor domain adaptation for high spatial resolution urban land-cover mapping: From airborne to spaceborne imagery},
    journal = {Remote Sensing of Environment},
    volume = {277},
    pages = {113058},
    year = {2022},
    issn = {0034-4257},
    doi = {https://doi.org/10.1016/j.rse.2022.113058},
    url = {https://www.sciencedirect.com/science/article/pii/S0034425722001729},
    author = {Junjue Wang and Ailong Ma and Yanfei Zhong and Zhuo Zheng and Liangpei Zhang},
    }
```
LoveCS can be used for academic purposes only,
<font color="red"><b> and any commercial use is prohibited.</b></font>
<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">

<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>