MISSING_LIBRARY_MESSAGE = 'CUDA runtime libraries cannot be found on this machine. Please re-install DeepView and try again!'

try :
    from habitat.analysis import Device
    from habitat.analysis.metrics import Metric
    from habitat.analysis.predictor import Predictor
    from habitat.tracking.operation import OperationTracker
except ImportError as ie:
    import traceback
    traceback.print_exc()
    print(MISSING_LIBRARY_MESSAGE)

__version__ = '1.0.0'
__description__ = 'Cross-GPU performance predictions for PyTorch neural network training.'

__author__ = 'Geoffrey Yu'
__email__ = 'gxyu@cs.toronto.edu'

__license__ = 'Apache-2.0'

__all__ = [
    'Device',
    'Metric',
    'OperationTracker',
    'Predictor',
]
