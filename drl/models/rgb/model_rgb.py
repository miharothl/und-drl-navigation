import torch
import torch.nn as nn
import torch.nn.functional as F

#
class QNetwork2(nn.Module):

    def __init__(self, h, w, channels, action_size, seed):
        super(QNetwork2, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, action_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
#



class QNetwork2a(nn.Module):

    def __init__(self, h, w, channels, action_size, seed):
        super(QNetwork2a, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # fc1: 512 = 64 * 4 * 4 units
        self.fc1 = nn.Linear(64 * 6 * 6, 512)

        # fc2: outputShape[0] units, one for each action
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)



#
#
class QNetwork3(nn.Module):
   """Actor (Policy) Model."""

   def __init__(self, state_size, state_size_2, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork3, self).__init__()
        nfilters = [128, 128*2, 128*2]
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv3d(3, nfilters[0], kernel_size=(1, 3, 3), stride=(1,3,3))
        self.bn1 = nn.BatchNorm3d(nfilters[0])
        self.conv2 = nn.Conv3d(nfilters[0], nfilters[1], kernel_size=(1, 3, 3), stride=(1,3,3))
        self.bn2 = nn.BatchNorm3d(nfilters[1])
        self.conv3 = nn.Conv3d(nfilters[1], nfilters[2], kernel_size=(4, 3, 3), stride=(1,3,3))
        self.bn3 = nn.BatchNorm3d(nfilters[2])
        conv_out_size = self._get_conv_out_size(state_size)
        fc = [conv_out_size, 1024]
        self.fc1 = nn.Linear(fc[0], fc[1])
        self.fc2 = nn.Linear(fc[1], action_size)


   def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self._cnn(state)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

   # generate input sample and forward to get shape
   def _get_conv_out_size(self, shape):
        x = torch.rand(shape)
        x = self._cnn(x)
        n_size = x.data.view(1, -1).size(1)
        print('Convolution output size:', n_size)
        return n_size

   def _cnn(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x
