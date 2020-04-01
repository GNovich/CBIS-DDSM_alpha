from torchvision import models
from torch.nn import Conv2d, Linear, Sequential, Softmax
from string import digits


class PreBuildConverter:
    def __init__(self, in_channels, out_classes, add_soft_max=True):
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.soft_max = add_soft_max

    def get_by_str(self, name):
        name_clean = name.translate(str.maketrans('', '', digits)).lower()
        if 'vgg' in name_clean:
            return self.VGG(name)
        if 'dense' in name_clean:
            return self.DenseNet(name)
        if 'mobilenet' in name_clean:
            return self.MobileNet()
        if 'resnet' in name_clean:
            return self.ResNet(name)

    def VGG(self, name='vgg16'):
        model = getattr(models, name)()
        conv = model.features[0]
        classifier = model.classifier[-1]

        model.features[0] = Conv2d(self.in_channels, conv.out_channels,
                                   kernel_size=conv.kernel_size, stride=conv.stride,
                                   padding=conv.padding, bias=conv.bias)
        model.classifier[-1] = Linear(in_features=classifier.in_features,
                                      out_features=self.out_classes, bias=True)

        return Sequential(model, Softmax(1)) if self.soft_max else model

    def DenseNet(self, name='densenet121'):
        model = getattr(models, name)()
        conv = model.features[0]
        classifier = model.classifier

        model.features[0] = Conv2d(self.in_channels, conv.out_channels,
                                   kernel_size=conv.kernel_size, stride=conv.stride,
                                   padding=conv.padding, bias=conv.bias)
        model.classifier = Linear(in_features=classifier.in_features,
                                  out_features=self.out_classes, bias=True)

        return Sequential(model, Softmax(1)) if self.soft_max else model

    def MobileNet(self):
        model = getattr(models, 'mobilenet_v2')()
        conv = model.features[0][0]
        classifier = model.classifier[-1]

        model.features[0][0] = Conv2d(self.in_channels, conv.out_channels,
                                      kernel_size=conv.kernel_size, stride=conv.stride,
                                      padding=conv.padding, bias=conv.bias)
        model.classifier[-1] = Linear(in_features=classifier.in_features,
                                      out_features=self.out_classes, bias=True)

        return Sequential(model, Softmax(1)) if self.soft_max else model

    def ResNet(self, name='resnet50'):
        model = getattr(models, name)()
        conv = model.conv1
        classifier = model.fc

        model.conv1 = Conv2d(self.in_channels, conv.out_channels,
                                      kernel_size=conv.kernel_size, stride=conv.stride,
                                      padding=conv.padding, bias=conv.bias)
        model.fc = Linear(in_features=classifier.in_features,
                                      out_features=self.out_classes, bias=True)

        return Sequential(model, Softmax(1)) if self.soft_max else model