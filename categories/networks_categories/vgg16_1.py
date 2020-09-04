import torch
import sys
sys.path.insert(0, '..')
import myutils
import torchvision.models as models
from torch import nn
from torchsummary import summary


def vgg16_1(unfreeze=False):
    """
    Unfreeze(True) all the model weights.
    Freeze(False) the convolutional layers only.
    """

    model = models.vgg16(pretrained=True)

    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
                        nn.Linear(num_ftrs, 304),
                        nn.ReLU(True),
                        nn.Dropout(p=0.1)
                        )

    class Net(nn.Module):


        def __init__(self):
            super().__init__()
            self.model = model
            nn.init.xavier_normal_(self.model.classifier[0].weight)


        def forward(self, x):
            x = self.model(x)
            return x

    net = Net()

    for param in model.parameters():
        param.requires_grad = unfreeze
    for param in model.classifier.parameters():
        param.requires_grad = True
    return net


def loss_fn(weight):
    #criterion = nn.MultiMarginLoss(weight=weight)
    criterion = nn.CrossEntropyLoss(weight=weight)
    return criterion


if __name__ == '__main__':
    net = vgg16_1()
    print(net)
    total_params, total_trainable_params = myutils.get_num_parameters(net)
    print('Total: {:,}\tTrainable: {:,}'.format(total_params, total_trainable_params))
    # To print a summary
    summary(net, (3, 224, 224))
