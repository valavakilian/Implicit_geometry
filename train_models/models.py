import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TwoLayerLinear(nn.Module):
  def __init__(self, n, d, k, bias=False):
    super(TwoLayerLinear, self).__init__()
    self.fc1 = nn.Linear(n, d, bias=bias)
    self.fc2 = nn.Linear(d, k, bias=bias)

  def forward(self, x):
    features = self.fc1(x)
    logits = self.fc2(features)
    return logits

  def get_features(self, x, device, torch_var=False):
    if torch_var:
      return self.fc1(x).to(device).detach()
    else:
      return self.fc1(x).to(device).detach().numpy().copy()

  def get_weights(self, device, torch_var=False):
    if torch_var:
      return self.fc2.weight.to(device).detach()
    else:
      return self.fc2.weight.to(device).detach().numpy().copy()


class MLP(nn.Module):
    def __init__(self, hidden, depth = 6, fc_bias = True, num_classes = 10):
        # Depth means how many layers before final linear layer
        super(MLP, self).__init__()
        layers = [nn.Linear(784, hidden, bias = False), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]
        for i in range(depth - 1):
            layers += [nn.Linear(hidden, hidden, bias = False), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, num_classes, bias = fc_bias)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        x = self.fc(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, K, input_ch):
        # Following same model setup of Papyan et al [2020]
        super(ResNet18, self).__init__()
        self.core_model = models.resnet18(pretrained=False, num_classes=K)
        self.core_model.conv1 = nn.Conv2d(input_ch, self.core_model.conv1.weight.shape[0], 3, 1, 1, bias=False) # Small dataset filter size used by He et al. (2015)
        self.core_model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.core_model.fc = nn.Linear(in_features=512, out_features=K, bias=False)
    
    def forward(self, x):
        return self.core_model(x)