from FCN.layers.blocks import vgg_block
from FCN.backbone.VGG import VGG16
from tensorflow.keras.layers import (
    Conv2D, Cropping2D, add,
    Conv2DTranspose, Activation
)
from tensorflow.keras.models import Model


class ConvModel:

    def __init__(self, input_shape = (512, 512, 3), output_channels = 1, backbone = 'vgg16', pretrained = True):
        '''
        Fully Convolutional Network
        
        Parameters:
            - input_shape       : Shape of input placeholder (tuple)
            - output_channels   : Number of channels of the output mask
            - backbone          : Type of Backbone
            - pretrained        : Use pretrained backbone or not
        '''
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.backbone = backbone
        self.pretrained = pretrained
        self.build_network()
    
    def build_backbone(self):
        '''
        Build Backbone for FCN
        '''
        if self.backbone == 'vgg16':
            self.backbone_model = VGG16(self.input_shape, self.pretrained)
            self.input_placeholder = self.backbone_model.input_placeholder
            self.output_placeholder = self.backbone_model.output_placeholder
    
    def build_network(self):
        '''
        Build FCN model
        '''
        self.build_backbone()
        if self.backbone == 'vgg16':
            x = self.output_placeholder
            
            x = Conv2D(4096, kernel_size = (7, 7), padding = 'same', name = 'fc6')(x)
            x = Activation('relu')(x)
            
            x = Conv2D(4096, kernel_size = (1, 1), padding = 'same', name = 'fc7')(x)
            x = Activation('relu')(x)
            
            x = Conv2D(self.output_channels, kernel_size = (1, 1), padding = 'same', name = 'score')(x)
            x = Activation('relu')(x)

            self.model = Model(self.input_placeholder, x)

            conv_size = self.model.layers[-1].output_shape[2]
            x = Conv2DTranspose(
                self.output_channels,
                kernel_size = (4, 4),
                strides = (2, 2),
                padding = 'valid',
                name = 'score_2'
            )(x)
            self.model = Model(self.input_placeholder, x)

            deconv_size = self.model.layers[-1].output_shape[2]
            extra = deconv_size - 2 * conv_size

            self.output_placeholder = Cropping2D(cropping = ((0, extra), (0, extra)))(x)
            self.model = Model(self.input_placeholder, self.output_placeholder)

            conv_size = self.model.layers[-1].output_shape[2]
            skip_conv_1 = Conv2D(self.output_channels, kernel_size = (1, 1), padding = 'same', name = 'score_pool4')
            concatenated = add([
                skip_conv_1(self.model.layers[14].output),
                self.model.layers[-1].output
            ])

            x = Conv2DTranspose(
                self.output_channels,
                kernel_size = (4, 4),
                strides = (2, 2),
                padding = 'valid',
                name = 'score_4'
            )(concatenated)
            x = Cropping2D(cropping = ((0, 2), (0, 2)))(x)

            skip_conv_2 = Conv2D(self.output_channels, kernel_size = (1, 1), padding = 'same', name = 'score_pool3')
            concatenated = add([skip_conv_2(self.model.layers[10].output), x])

            x = Conv2DTranspose(
                self.output_channels,
                kernel_size = (16, 16),
                strides = (8, 8),
                padding = 'valid',
                name = 'Final_upsample'
            )(concatenated)
            self.output_placeholder = Cropping2D(cropping = ((0, 8), (0, 8)))(x)

            self.model = Model(self.input_placeholder, self.output_placeholder)

    
    def summarize(self):
        '''
        Print summary of FCN model
        '''
        self.model.summary()