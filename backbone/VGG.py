from tensorflow.keras.applications.vgg16 import VGG16 as vgg_pretrained

class VGG16:

    def __init__(self, input_shape, pretrained):
        self.input_shape = input_shape
        self.pretrained = pretrained
        self.build_backbone()
    
    def build_backbone(self):
        if self.pretrained:
            self.backbone = vgg_pretrained(weights = 'imagenet', include_top = False, input_shape = self.input_shape)
            self.backbone.trainable = False
        else:
            self.backbone = vgg_pretrained(weights = None, include_top = False, input_shape = self.input_shape)
        self.input_placeholder = self.backbone.input
        self.output_placeholder = self.backbone.output