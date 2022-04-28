from data.cross_data import CSLoader
import logging
logger = logging.getLogger(__name__)
from utils.tools import *
from utils.tta import *
from module.csn import change_csn, replace_bn_with_csn
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run lovecs methods.')

parser.add_argument('--config_path',  type=str,
                    help='config path', default='sfpn')
parser.add_argument('--ckpt_path',  type=str,
                    help='weights path', default='./log/sfpn.pth')
args = parser.parse_args()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def evaluate_cs(model, cfg, is_training=False, ckpt_path=None, logger=None):

    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = er.viz.VisualizeSegmm(vis_dir, palette)
    if not is_training:
        model_state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_state_dict,  strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)

    model.eval()
    print(cfg.EVAL_DATA_CONFIG)
    eval_dataloader = CSLoader(cfg.EVAL_DATA_CONFIG)
    metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)

    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(device)
            cls = tta(model, ret.to(device), tta_config=cfg.TTA_LIST)
            cls = cls.argmax(dim=1).cpu().numpy()
            cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)
            mask = cls_gt >= 0
            y_true = cls_gt[mask].ravel()
            y_pred = cls[mask].ravel()

            metric_op.forward(y_true, y_pred)

            if cfg.SNAPSHOT_DIR is not None:
                for fname, pred in zip(ret_gt['fname'], cls):
                    viz_op(pred, fname.replace('tif', 'png'))

    metric_op.summary_all()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    seed_torch(2333)
    from module.semantic_fpn import SemanticFPN
    cfg = import_config(args.config_path)
    # Semantic Segmentation model
    model = SemanticFPN(**cfg.MODEL).cuda()
    # Replace the BNs into CSNs
    model = replace_bn_with_csn(model)
    change_csn(model, source=False)
    evaluate_cs(model, cfg, False, args.ckpt_path, logger)