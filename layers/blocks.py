from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose


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