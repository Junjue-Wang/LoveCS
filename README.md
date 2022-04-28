<h2 align="center">Cross-sensor domain adaptation for high-spatial resolution urban land-cover mapping: from airborne to spaceborne imagery</h2>


<h5 align="right">by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>, 
Ailong Ma,
<a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>, 
<a href="http://zhuozheng.top/">Zhuo Zheng</a>, and Liangpei Zhang</h5>

[[`Paper`](https://www.researchgate.net/profile/Junjue-Wang/publication/355390292_LoveDA_A_Remote_Sensing_Land-Cover_Dataset_for_Domain_Adaptive_Semantic_Segmentation/links/617cd8bda767a03c14cecbc9/LoveDA-A-Remote-Sensing-Land-Cover-Dataset-for-Domain-Adaptive-Semantic-Segmentation.pdf?_sg%5B0%5D=Iw5FPui1-9iYrZN7aZO766hZA-LmublHlq8bp0694vUeIGDIzp5SGTfYN-OWhurZOujSPU0RDZ5lW0i02HVUew.7x9qdrvJwRmAnsqEyh5-xSFdh0M9AaTpdXcZCfHyhVl5GNLR5nlDIx8ctTXFy1HE1yNexX4ytzYqJWkAGJVTvg.Rrg3rXhcp9mMlLTU3n9Jf-h0Kt8VzHAd0AmhG2yPQxI-yRK6J0wAulUZ65dih6BQ9CbrQm0_23_nULO_BXwaJg&_sg%5B1%5D=KLu7pn0g50f8FwKE9x5iOuDPYb8VaOpX4k_ieq8eWJVVeJyXbZJO-O4pCL687QRxYbBnWdo7fJj8FZEOc3t3lgVVyDz0CFS-ff7LToXj4R9W.7x9qdrvJwRmAnsqEyh5-xSFdh0M9AaTpdXcZCfHyhVl5GNLR5nlDIx8ctTXFy1HE1yNexX4ytzYqJWkAGJVTvg.Rrg3rXhcp9mMlLTU3n9Jf-h0Kt8VzHAd0AmhG2yPQxI-yRK6J0wAulUZ65dih6BQ9CbrQm0_23_nULO_BXwaJg&_iepl=)],
[[`BibTex`](https://slideslive.com/38969542)],
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
If you use LoveDA in your research, please cite our coming NeurIPS2021 paper.
```text
    @inproceedings{wang2021loveda,
        title={Love{DA}: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation},
        author={Junjue Wang and Zhuo Zheng and Ailong Ma and Xiaoyan Lu and Yanfei Zhong},
        booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
        editor = {J. Vanschoren and S. Yeung},
        year={2021},
        volume = {1},
        pages = {},
        url={https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf}
    }
```
The LoveCS can be used for academic purposes only,
<font color="red"><b> and any commercial use is prohibited.</b></font>
<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">

<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>