from torch import nn


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000, num_layers=1):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels

        if num_layers == 1:
            self.linear = nn.Linear(dim, num_labels)
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()
        else:
            linear_layers = list()
            linear_layers.append(nn.Linear(dim, 1000))
            linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Linear(1000, num_labels))
            linear_layers[-1].data.normal_(mean=0.0, std=0.01)
            linear_layers[-1].bias.data.zero_()
            self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
