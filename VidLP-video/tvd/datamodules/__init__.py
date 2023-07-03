from .drama_datamodule import DramaMDataModule
from .drama_ft_datamodule import DramaFTMDataModule
from .drama_pt_datamodule import DramaPTMDataModule

_datamodules = {"drama": DramaMDataModule, "dramaFT": DramaFTMDataModule, "dramaPT": DramaPTMDataModule}
