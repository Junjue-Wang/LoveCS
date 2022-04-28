import torch
import torch.nn as nn
from ever.interface import ERModule
from ever import registry
from module.base import AssymetricDecoder, FPN, default_conv_block
from segmentation_models_pytorch.encoders import get_encoder

@registry.MODEL.register('SemanticFPN')
class SemanticFPN(ERModule):
    def __init__(self, config):
        super(SemanticFPN, self).__init__(config)
        self.en = get_encoder(**self.config.encoder)
        self.fpn = FPN(**self.config.fpn)
        self.decoder = AssymetricDecoder(**self.config.decoder)
        self.cls_pred_conv = nn.Conv2d(self.config.decoder.out_channels, self.config.classes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    def forward(self, x, y=None):
        feat_list = self.en(x)[1:]
        fpn_feat_list = self.fpn(feat_list)
        final_feat = self.decoder(fpn_feat_list)
        cls_pred = self.cls_pred_conv(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)
        return cls_pred



    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(
                name='resnet50',
                weights='imagenet',
                in_channels=3
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                conv_block=default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            classes=7,
        ))


