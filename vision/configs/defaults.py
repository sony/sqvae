from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)
_C.path = "/checkpoint_path" # To be set in advance
_C.path_dataset = "/dataset_path" # To be set in advance
_C.nworker = 2
_C.list_dir_for_copy = ['', 'networks/'] # []


_C.dataset = CN(new_allowed=True)

_C.model = CN(new_allowed=True)

_C.network = CN(new_allowed=True)

_C.train = CN(new_allowed=True)
_C.train.bs = 32
_C.train.lr = 0.001
_C.train.epoch_max = 100

_C.quantization = CN(new_allowed=True)
_C.quantization.temperature = CN(new_allowed=True)
_C.quantization.temperature.init = 1.0
_C.quantization.temperature.decay = 0.00001
_C.quantization.temperature.min = 0.0

_C.test = CN(new_allowed=True)
_C.test.bs = 50

_C.flags = CN(new_allowed=True)
_C.flags.arelbo = True
_C.flags.decay = True
_C.flags.bn = True


def get_cfgs_defaults():
  return _C.clone()
