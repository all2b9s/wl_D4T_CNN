"""Training losses of src.train_d4, one class per term.

Data terms (share the trainer's microbatch forward):
    SupervisedLoss(scale)(shape, truth, w, count)
    ScoreLoss(isotropy_weight, ema_decay)(shape, response, w, dw, group, count)
Auxiliary terms (own forward on noiseless isolated partners; AuxLoss interface):
    RotationLoss, NonlinearLoss, PositionAliasLoss, PositionNullLoss
"""
from src.loss.common import set_response_mode
from src.loss.base import AuxLoss
from src.loss.supervised_loss import SupervisedLoss
from src.loss.score_SSL import ScoreLoss
from src.loss.rot_loss import RotationLoss
from src.loss.nolinear_loss import NonlinearLoss
from src.loss.position_loss import PositionAliasLoss, PositionNullLoss
