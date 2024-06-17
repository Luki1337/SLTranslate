import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

def create_network(freeze_pretrained=False):
    # Load pretrained model
    net = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

    # Freeze all pretrained layers
    if freeze_pretrained:
        for param in net.parameters():
            param.requires_grad = False

    # Modify the final fully connected layer
    in_features = 1408  # Number of features of the classifier
    out_features = 26  # Number of output classes
    net.classifier[-1] = nn.Linear(in_features,  out_features, bias=True)

    # Create a list of parameters to be optimized
    parameters_to_optimize = filter(lambda p: p.requires_grad, net.parameters())

    return net, parameters_to_optimize

