"""
Multi-class Geometric Viewpoint Aware Loss
A PyTorch implmentation of the geometric-aware softmax view loss as described in
RenderForCNN (link: https://arxiv.org/pdf/1505.05641.pdf)
Caffe implmentation:
https://github.com/charlesq34/caffe-render-for-cnn/blob/view_prediction/
"""
import torch
from torch      import nn
import torch.nn.functional  as F
import numpy                as np

# pylint: disable=no-self-use

class SoftmaxVPLoss(nn.Module):
    """A class for the softmax viewpoint loss from the R4CNN paper.
    """
    # Loss parameters taken directly from Render4CNN paper
    # azim_band_width = 7     # 15 in paper
    # elev_band_width = 2     # 5 in paper
    # tilt_band_width = 2     # 5 in paper

    # azim_sigma = 5
    # elev_sigma = 3
    # tilt_sigma = 3
    def __init__(self, kernel_size=7, sigma=25):
        """Initialize the loss

        Args:
            kernel_size: The size of the kernel.
            sigma: The stdev of the kernel.

        Returns:
            Nothing.
        """
        super(SoftmaxVPLoss, self).__init__()
        self.filter = self.viewloss_filter(kernel_size, sigma)
        self.kernel_size = kernel_size

    def viewloss_filter(self, size, sigma):
        """Creates the filter which is convolved with the one-hot.

        Args:
            size: The width of this filter.
            sigma: The stdev of this filter.

        Returns:
            The new filter.
        """
        vec = np.linspace(-1*size, size, 1 + 2*size, dtype=np.float)
        prob = np.exp(-1 * abs(vec) / sigma)
        # normalize filter because otherwise loss will scale with kernel size
        prob = torch.FloatTensor(prob)[None, None, :]    # 1 x 1 x (1 + 2*size)
        return prob


    def forward(self, preds, labels, size_average=True, do_reduce=True):
        """Perform a forward pass of the loss.

        Args:
            preds: Angle predictions (batch_size, 360 x num_classes)
            targets: Angle labels (batch_size, 360 x num_classes)

        Returns:
            The calculated loss.
        """
        # Set absolute minimum for numerical stability
        # (assuming float16 - 6x10^-5)
        assert len(labels.shape) == 1
        batch_size = labels.shape[0]

        # Construct one-hot labels (dimension is (batch x 1 x dimension))

        #I thought that creation of a new tensor might slow down calculations
        #but it doesn't seem to slow things;
        #10^4 iterations of scatter to batch of 256 took 0.5 sec
        #
        #speed test:
        #    scatter is ~25% faster for small batches (32),
        #    indexing is ~25% faster for larger batches (>1024)
        #    both are the same around 256 batch size

        labels = labels.long()
        labels_oh = torch.zeros(batch_size, 360)
        labels_oh[torch.arange(batch_size), labels] = 1.

        # Concat one hot vector and convolve
        labels_oh = torch.cat((labels_oh[:, -self.kernel_size:],
                               labels_oh,
                               labels_oh[:, :self.kernel_size]),
                              dim=1)

        labels_oh = F.conv1d(labels_oh[:, None, :], self.filter)

        # convert labels to CUDA
        labels_oh = labels_oh.squeeze(1).cuda()

        # calculate loss -- sum from paper
        loss = (-1*labels_oh*preds.log_softmax(1)).sum(1)

        if not do_reduce:
            pass
        elif size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss
