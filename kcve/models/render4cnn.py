"""Model for render4cnn."""
import torch
import torch.nn as nn

# pylint: disable=C0103,
class render4cnn(nn.Module):
    """Model for render4cnn."""
    def __init__(self, weights_path=None):
        super(render4cnn, self).__init__()

        # define model
        self.conv4 = nn.Sequential(
                        nn.Conv2d(3, 96, (11, 11), (4, 4)),
                        nn.ReLU(),
                        nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
                        nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.),
                        nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2), 1, 2),
                        nn.ReLU(),
                        nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
                        nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.),
                        nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)),
                        nn.ReLU(),
                        nn.Conv2d(384, 384, (3, 3), (1, 1), (1, 1), 1, 2),
                        nn.ReLU())

        self.conv5 = nn.Sequential(
                        nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1), 1, 2),
                        nn.ReLU(),
                        nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True))

        self.infer = nn.Sequential(
                        nn.Linear(9216, 4096),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(4096, 4096),
                        nn.ReLU(),
                        nn.Dropout(0.5))

        self.azim = nn.Linear(4096, 12*360)
        self.elev = nn.Linear(4096, 12*360)
        self.tilt = nn.Linear(4096, 12*360)

        if weights_path is not None:
            self._initialize_weights(weights_path)

    # weight initialization from torchvision/models/vgg.py
    def _initialize_weights(self, weights_path):
        """Initialize the R4CNN weights.

        Args:
            The paths to the weights.

        Output:
            None.
        """
        self.load_state_dict(torch.load(weights_path))

    def forward(self, x, obj_class):
        """Perform a forward pass.

        Args:
            x: A batch of vehicle crops.
            obj_class: The object class.

        Returns:
            Azimuth, elevation, and tilt outputs.
        """
        # generate output
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.infer(x)

        # mask on class
        azim = self.azim(x)
        azim = azim.view(-1, 12, 360)
        azim = azim[torch.arange(x.shape[0]), obj_class, :]
        elev = self.elev(x)
        elev = elev.view(-1, 12, 360)
        elev = elev[torch.arange(x.shape[0]), obj_class, :]
        tilt = self.tilt(x)
        tilt = tilt.view(-1, 12, 360)
        tilt = tilt[torch.arange(x.shape[0]), obj_class, :]

        return azim, elev, tilt
