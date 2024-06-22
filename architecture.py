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

        self.most_similar_indices = {'conv1': None, 'conv2': None, 'conv4': None}

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

    def find_most_similar_filters_old(self, filters, filters_conv3):
        # Initialize lists to store the indices of most similar filters
        most_similar_indices = []
        mse_loss = nn.MSELoss(reduction='mean')

        for filter in filters:
            for kernel in filter:
                min_loss = float('inf')
                min_filter_index = -1
                min_kernel_index = -1
                for filt_idx, conv3_filter in enumerate(filters_conv3):
                    for kern_idx, c3_kernel in enumerate(conv3_filter):
                        mse = mse_loss(kernel, c3_kernel)
                        if mse < min_loss:
                            min_loss = mse
                            min_filter_index = filt_idx
                            min_kernel_index = kern_idx
                most_similar_indices.append((min_filter_index, min_kernel_index))

        return torch.tensor(most_similar_indices)

    def find_most_similar_filters(self, filters, filters_conv3, batch_size=16):
        # Initialize lists to store the indices of most similar filters
        most_similar_indices = []

        num_filters = filters.shape[0]
        num_conv3_filters = filters_conv3.shape[0]

        # Process in batches to reduce memory usage
        for i in range(0, num_filters, batch_size):
            batch_filters = filters[i:i + batch_size]
            batch_filters = batch_filters.view(-1, 1, *batch_filters.shape[2:])  # Shape: [batch_size*num_kernels, 1, 3, 3]

            min_loss_indices_batch = []

            for j in range(0, num_conv3_filters, batch_size):
                batch_filters_conv3 = filters_conv3[j:j + batch_size]
                batch_filters_conv3 = batch_filters_conv3.view(1, -1, *batch_filters_conv3.shape[2:])  # Shape: [1, batch_size*num_kernels_conv3, 3, 3]

                # Calculate the MSE for all kernel pairs at once using broadcasting
                mse_losses = ((batch_filters - batch_filters_conv3) ** 2).mean(dim=(-1, -2))  # Shape: [batch_size*num_kernels, batch_size*num_kernels_conv3]

                # Find the minimum MSE values and their corresponding indices
                min_loss_indices = torch.argmin(mse_losses, dim=1)  # Shape: [batch_size*num_kernels]
                min_loss_indices_batch.append(min_loss_indices)

            min_loss_indices_batch = torch.cat(min_loss_indices_batch, dim=0)
            for idx in min_loss_indices_batch:
                filter_index = idx // filters_conv3.size(1)
                kernel_index = idx % filters_conv3.size(1)
                most_similar_indices.append((filter_index.item(), kernel_index.item()))

        return torch.tensor(most_similar_indices)

    def compute_filter_similarity_loss(self):
        # Get filters from conv3
        filters_conv3 = self.conv3.weight  # shape: [512, 256, 3, 3]

        # Get filters from conv1, conv2, and conv4
        filters_conv1 = self.conv1.weight  # shape: [128, 3, 3, 3]
        filters_conv2 = self.conv2.weight  # shape: [256, 128, 3, 3]
        filters_conv4 = self.conv4.weight  # shape: [128, 512, 3, 3]

        # Find and store the indices of the most similar filters
        if self.most_similar_indices['conv1'] is None:
            self.most_similar_indices['conv1'] = self.find_most_similar_filters(filters_conv1, filters_conv3)
        if self.most_similar_indices['conv2'] is None:
            self.most_similar_indices['conv2'] = self.find_most_similar_filters(filters_conv2, filters_conv3)
        if self.most_similar_indices['conv4'] is None:
            self.most_similar_indices['conv4'] = self.find_most_similar_filters(filters_conv4, filters_conv3)
        print("Found all similarity indices")
        mse_loss = nn.MSELoss()

        # Compute MSE losses
        import pdb;pdb.set_trace()
        sim_loss1 = mse_loss(filters_conv1, filters_conv3[self.most_similar_indices['conv1']])
        sim_loss2 = mse_loss(filters_conv2, filters_conv3[self.most_similar_indices['conv2']])
        sim_loss4 = mse_loss(filters_conv4, filters_conv3[self.most_similar_indices['conv4']])

        # Aggregate similarity losses
        total_sim_loss = (sim_loss1 + sim_loss2 + sim_loss4) / 3
        
        return total_sim_loss


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