import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, InputSpec,
    MaxPooling2D, Layer,
    Conv2DTranspose
)
from tensorflow.keras import backend as K
from tensorflow.keras.utils import normalize


def vgg_block(input_tensor, n_conv_layer, filters, kernel_size, activation, block):
        '''
        Returns a VGG16 Block of the structure:
            conv_layer * n_conv_layer -> pooling_layer
        
        Parameters:
            - input_tensor  : Tensor
            - n_conv_layer  : Number of convolutional layers
            - filters       : Number of filters in the convolutional layer
            - kernel_size   : Size of convolutional kernels
            - activation    : Activation Function (str or Layer)
            - block         : Block number
        
        Output:
            - Tensor
        '''
        x = Conv2D(
            filters, kernel_size,
            activation = activation,
            padding = 'same',
            name = 'block' + str(block) + '_conv1'
        )(input_tensor)
        for i in range(n_conv_layer - 1):
            x = Conv2D(
                filters, kernel_size,
                activation = activation,
                padding = 'same',
                name = 'block' + str(block) + '_conv' + str(i + 2)
            )(x)
        x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block' + str(block) + '_pool')(x)
        return x


def resize_images(x, size, method = 'bilinear'):
    new_size = tf.convert_to_tensor(size, dtype = tf.int32)
    resized = tf.image.resize_images(x, new_size)
    return resized


class BilinearUpSampling2D(Layer):
    '''
    Upsampling2D with bilinear interpolation
    '''

    def __init__(self, target_shape = None, data_format = None, **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {
            'channels_last',
            'channels_first'
        }
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim = 4)]
        self.target_shape = target_shape
        if self.data_format == 'channels_first':
            self.target_size = (target_shape[2], target_shape[3])
        elif self.data_format == 'channels_last':
            self.target_size = (target_shape[1], target_shape[2])
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (
                input_shape[0],
                self.target_size[0],
                self.target_size[1],
                input_shape[3]
            )
        else:
            return (
                input_shape[0],
                input_shape[1],
                self.target_size[0],
                self.target_size[1]
            )

    def call(self, inputs):
        return resize_images(
            inputs,
            size = self.target_size,
            method = 'bilinear'
        )

    def get_config(self):
        config = {
            'target_shape': self.target_shape,
            'data_format': self.data_format
        }
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))