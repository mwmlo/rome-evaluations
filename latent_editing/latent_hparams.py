from dataclasses import dataclass

from util.hparams import HyperParams

@dataclass
class LatentHyperParams(HyperParams):
    model_name: str
    n_epochs: int
    overwrite: bool
