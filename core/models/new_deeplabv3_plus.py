from torchvision import models
from base_models.efficientnet import efficientnet
from base_models.inceptionv3 import inceptionv3_backbone
from base_models.resnext import resnext50_32x4d

# Use DeepLabV3+ with EfficientNet as the backbone
def get_new_deeplabv3(backbone='resnet50', pretrained=False):
    weights = models.DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights)

    # Replace the ResNet backbone with EfficientNet
    if backbone == 'resnet50':
        pass
    elif backbone == 'efficient_net':
        back = efficientnet(pretrained=True)
        model.backbone = back
    elif backbone == 'inceptionv3':
        back = inceptionv3_backbone(pretrained=True)
        model.backbone = back
    elif backbone == 'resnext':
        back = resnext50_32x4d(pretrained=True)
        model.backbone = back
    else:
        raise("That backbone is not available")

    # Modify other components if needed
    # ...

    # Print the modified DeepLabV3+ model
    return model