"""Weakly Supervised Segmentation methods for medical images.

Retained methods (each maps to a documented paper, the implementation is
the version actually exercised by ``train_weakly_supervised.py``):

    - BoxSupervised : Box-only mask + foreground/background CE
                      (BoxSup / BoxInst family, MIL projection variant)
    - CAM           : Class Activation Mapping with Grad-CAM hooks
                      (Zhou et al., CVPR 2016 / Selvaraju et al., ICCV 2017)
    - MIL           : Multi-instance learning from image-level labels
    - Point         : Bearman et al., ECCV 2016 point supervision
    - TreeEnergy    : Tree-structured energy minimization
    - SEAM          : Wang et al., CVPR 2020 self-supervised equivariant attention
    - PuzzleCAM     : Jo & Yu, ICIP 2021 puzzle piece matching
    - AdvCAM        : Lee et al., CVPR 2021 adversarial complementary erasing
    - MCTformer     : Xu et al., CVPR 2022 multi-class token transformer
    - ScribbleSup   : Lin et al., CVPR 2016 scribble supervision (light variant
                      with an inlined differentiable pairwise CRF surrogate)

Methods that previously lived here (GrabCut, fBRS, RITM, SimpleClick,
iSeg, ClickSupervision, Affinity, GatedCRF, EMPseudoLabel, SAMGuidedWeak,
BACoN, WPGSeg, Scribble, SeCo, DuPL, CTI, WeCLIP, S2C, DiG, PCSS,
GazeMedSeg, SimTxtSeg, ExCEL, IRNet, AuxSeg) were removed because their
implementations diverged substantially from the originating papers
(missing GMM/graph-cut/feature back-prop/frozen CLIP/SAM/text encoders,
fabricated references, or faithful source-code references) and could not
be exercised without adding entire model branches and dataloader streams.
"""

from .cam_generator import CAMGenerator
from .box_supervised import BoxSupervisedLoss
from .cam import CAMLoss
from .mil import MILLoss
from .point_supervised import PointSupervisedLoss
from .tree_energy import TreeEnergyLoss
from .seam import SEAMLoss
from .puzzle_cam import PuzzleCAMLoss
from .adv_cam import AdvCAMLoss
from .mctformer import MCTformerLoss
from .eps import EPSLoss
from .boxinst import BoxInstLoss
from .scribble_sup import ScribbleSupLoss
from .recam import ReCAMLoss
from .toco import ToCoLoss
from .lpcam import LPCAMLoss
from .mars import MARSLoss
from .dupl import DuPLLoss
from .more import MoReLoss
from .psdpm import PSDPMLoss
from .semples import SemPLeSLoss

__all__ = [
    'CAMGenerator',
    'BoxSupervisedLoss',
    'CAMLoss',
    'MILLoss',
    'PointSupervisedLoss',
    'TreeEnergyLoss',
    'SEAMLoss',
    'PuzzleCAMLoss',
    'AdvCAMLoss',
    'MCTformerLoss',
    'EPSLoss',
    'BoxInstLoss',
    'ScribbleSupLoss',
    'ReCAMLoss',
    'ToCoLoss',
    'LPCAMLoss',
    'MARSLoss',
    'DuPLLoss',
    'MoReLoss',
    'PSDPMLoss',
    'SemPLeSLoss',
]
