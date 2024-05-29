from habitat.analysis.device import _Device

Device = _Device()

SPECIAL_OPERATIONS = {
    # Convolution
    'conv2d',
    'conv_transpose2d',

    # Matrix multiply operations
    'linear',
    '__matmul__', # calls the same kernel as linear
    'bmm',

    # Recurrent operations
    'lstm',
    'gru',
    'rnn_tanh',
    'rnn_relu',
}
