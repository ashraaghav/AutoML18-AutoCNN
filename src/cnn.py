import numpy as np
import torch.nn as nn


ACTIVATION_DICT = {'relu': nn.ReLU,
                   'sigmoid': nn.Sigmoid,
                   'tanh': nn.Tanh}


# define ConvNet #######################################################################################################
class ConfigurableNet(nn.Module):
    """
    Example of a configurable network. (No dropout or skip connections supported)
    """
    def _update_size(self, dim, padding, dilation, kernel_size, stride):
        """
        Helper method to keep track of changing output dimensions between convolutions and Pooling layers
        returns the updated dimension "dim" (e.g. height or width)
        """
        # Altering the equation because it was not working if MaxPool layers were not included.
        # Reference: output_size=(w+2*pad-(d(k-1)+1))/s+1
        # return int(np.floor((dim + 2 * padding - dilation * (kernel_size - 1) + 1) / stride))
        return int(np.floor((dim + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1))

    def __init__(self, config, num_classes=10, height=28, width=28, channels=1):
        """
        Configurable network for image classification
        :param config: network config to construct architecture with
        :param num_classes: Number of outputs required
        :param height: image height
        :param width: image width
        """
        super(ConfigurableNet, self).__init__()
        self.config = config

        # Keeping track of internals like changeing dimensions
        n_layers = config['n_conv_layers'] + config['n_full_layers']
        n_convs = config['n_conv_layers']
        conv_layer = 0
        self.layers = []
        self.mymodules = nn.ModuleList()
        out_channels = channels

        # Create sequential network
        # NOTE Making padding to initialize image with (32, 32) beacuse we don't want to lose information (padding+=2)
        # NOTE Keeping the 'same' padding instead to avoid downsampling (default was 2)
        for layer in range(n_layers):
            if n_convs >= 1:  # This way it only supports multiple convolutional layers at the beginning (not inbetween)
                l = []  # Conv layer can be sequential layer with Batch Norm and pooling
                stride = 1
                kernel_size = config['ckernel_size']
                padding = int(kernel_size/2)
                if layer == 0:
                    padding += 2
                dilation = 1  # fixed
                if conv_layer == 0:
                    out_channels = config['channels']
                else:
                    # instead of handling different widths for each conv layer, just per convolution add the same size
                    out_channels += config['channels']

                # get convolution
                c = nn.Conv2d(channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

                # update dimensions
                channels = out_channels
                height = self._update_size(height, padding, dilation, kernel_size, stride)
                width = self._update_size(width, padding, dilation, kernel_size, stride)
                l.append(c)

                # batchnorm yes or no?
                batchnorm = config['batch_norm']
                if batchnorm:
                    b = nn.BatchNorm2d(channels)
                    l.append(b)

                # # determine activation function
                act = ACTIVATION_DICT[config['conv_activation']]
                l.append(act())

                # do max pooling yes or no?
                max_pooling = True
                if max_pooling:
                    # get the correct max pool kernel from hyper parameter config
                    m_ks = m_stride = config['mkernel_size']
                    pool = nn.MaxPool2d(kernel_size=m_ks, stride=m_stride)
                    l.append(pool)
                    height = self._update_size(height, 0, 1, m_ks, m_stride)
                    width = self._update_size(width, 0, 1, m_ks, m_stride)
                n_convs -= 1
                conv_layer += 1

                # add dropout layer
                d2 = nn.Dropout2d(p=config['conv_dropout'])
                l.append(d2)

                # setup everything as sequential layer
                s = nn.Sequential(*l)
                self.mymodules.append(s)
                self.layers.append(s)

            # handle intermediate fully connected layers
            elif layer < n_layers-1:
                if n_convs == 0:  # compute fully connected input size
                    channels = height * width * channels
                    n_convs -= 1
                #           in_channels, out_channels
                lay = nn.Linear(channels, config['n_neurons'])
                self.mymodules.append(lay)
                self.layers.append(lay)
                channels = config['n_neurons']  # update channels to keep track how many inputs lead to the next layer

                # Add activation to intermediate layers, if given in config
                if config['full_activation'] != 'None':
                    act = ACTIVATION_DICT[config['full_activation']]
                    act = act()
                    self.mymodules.append(act)
                    self.layers.append(act)

                # add dropout layer
                d = nn.Dropout(p=config['conv_dropout'])
                self.mymodules.append(d)
                self.layers.append(d)

            # handle final fully connected layer
            else:
                if n_convs == 0:
                    channels = height * width * channels
                    n_convs -= 1
                out = nn.Linear(channels, num_classes)
                self.mymodules.append(out)
                self.layers.append(out)

    def forward(self, out):
        for idx, layer in enumerate(self.layers):
            if self.config['n_conv_layers'] == idx:
                out = out.reshape(out.size(0), -1)  # flatten the output after convolutions (keeping batch dimension)
            out = layer(out)
        return out
