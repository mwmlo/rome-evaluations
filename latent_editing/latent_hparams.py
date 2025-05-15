from dataclasses import dataclass

from util.hparams import HyperParams

@dataclass
class LatentHyperParams(HyperParams):
    n_epochs: int
    overwrite: bool
