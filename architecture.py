import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomposedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DecomposedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.filters = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))

    def forward(self, x):
        batch_size, _, height, width = x.size()
        output_height = (height - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        output_width = (width - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        output = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=x.device)
        
        for i in range(self.out_channels):
            filter_i = self.filters[i:i+1, :, :, :]
            output[:, i:i+1, :, :] = F.conv2d(x, filter_i, stride=self.stride, padding=self.padding)
        
        return output


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = DecomposedConv2d(inplanes, squeeze_planes, kernel_size=(1, 1))
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = DecomposedConv2d(squeeze_planes, expand1x1_planes, kernel_size=(1, 1))
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = DecomposedConv2d(squeeze_planes, expand3x3_planes, kernel_size=(3, 3), padding=(1, 1))
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squeeze_activation(x)
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class DecomposedSqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DecomposedSqueezeNet, self).__init__()
        self.features = nn.Sequential(
            DecomposedConv2d(3, 96, kernel_size=(7, 7), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            DecomposedConv2d(512, num_classes, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x
    


class SimpleConvTestNet(nn.Module):
    def __init__(self):
        super(SimpleConvTestNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, stride=2, bias=False)
        import pdb;pdb.set_trace()
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 256, 3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 512, 3, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(512, 128, 3, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)





class SimpleConvTestNetDecomp(nn.Module):
    def __init__(self):
        super(SimpleConvTestNetDecomp, self).__init__()
        self.conv1 = DecomposedConv2d(3, 128, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = DecomposedConv2d(128, 256, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.conv3 = DecomposedConv2d(256, 512, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU()
        self.conv4 = DecomposedConv2d(512, 128, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)