from deepclustering2.arch import _register_arch
from .unet import UNet, FeatureExtractor as UNetFeatureExtractor
from .unet_convlstm import LSTM_Corrected_Unet
_register_arch("ContrastUnet", UNet)
