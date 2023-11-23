from torchvision import models
from base_models.efficientnet import efficientnet
from base_models.inceptionv3 import inceptionv3_backbone
from base_models.resnext import resnext50_32x4d

<<<<<<< HEAD
__all__ = ['get_new_deeplabv3']
# Use DeepLabV3+ with EfficientNet as the backbone
def get_new_deeplabv3(backbone='resnet50', pretrained=True):
    if pretrained:
        weights = models.DeepLabV3_ResNet50_Weights.DEFAULT
        model = models.segmentation.deeplabv3_resnet50(weights = weights)
    else:
        model = models.segmentation.deeplabv3_resnet50()
    # Replace the ResNet backbone with others
    if backbone == 'efficient_net':
=======
# Use DeepLabV3+ with EfficientNet as the backbone
def get_new_deeplabv3(backbone='resnet50', pretrained=False):
    weights = models.DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights)

    # Replace the ResNet backbone with EfficientNet
    if backbone == 'resnet50':
        pass
    elif backbone == 'efficient_net':
>>>>>>> 03b15e316cf37e6f6aaa7ece26625b1f20352a7d
        back = efficientnet(pretrained=True)
        model.backbone = back
    elif backbone == 'inceptionv3':
        back = inceptionv3_backbone(pretrained=True)
        model.backbone = back
    elif backbone == 'resnext':
        back = resnext50_32x4d(pretrained=True)
        model.backbone = back
    else:
<<<<<<< HEAD
        raise ValueError("That backbone is not available")
=======
        raise("That backbone is not available")
>>>>>>> 03b15e316cf37e6f6aaa7ece26625b1f20352a7d

    # Modify other components if needed
    # ...

    # Print the modified DeepLabV3+ model
    return model