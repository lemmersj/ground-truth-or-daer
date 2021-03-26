"""Reject model for click-here cnn.
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models import clickhere_cnn, render4cnn
from util import Paths

NUM_KEYPOINTS = 34

class RejectModel(nn.Module):
    """The model that performs the rejection.

    This is done by attempting to regress the additional error given the
    primary input (image) and seed (keypoint click, represented as a
    chebyshev distance map). A higher additional error indicates that it
    should reject.

    Attributes:
        cx_loss_fn: The loss function (crossentropy)
        binary_loss_fn: The loss function (BCE)
        model: the clickhere cnn model we use as a base.
        extra_linear: the linear layers that go from the chcnn embedding to
        the correct output format.
    """
    def __init__(self, num_bins):
        """ Initialize the rejection model.

        Args:
            num_bins: the number of output bins.

        Returns:
            Nothing.
        """
        # Initialize the superclass.
        super().__init__()

        # Initialize the loss function
        self.cx_loss_fn = nn.CrossEntropyLoss(
            reduction='none')
        self.binary_loss_fn = nn.BCELoss(
            reduction='none')

        # Initialize the clickhere-cnn starting model. Note it is
        # pretrained.
        self.model = clickhere_cnn(
            render4cnn(), weights_path=Paths.clickhere_weights)

        # And add the randomly initialized linear layers.
        # There is a set of bins for each keypoint, plus one for the "does it
        # make a difference?" prediction.
        self.extra_linear = nn.Sequential(
            nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(),\
            nn.Linear(4096, NUM_KEYPOINTS*(num_bins+1)))

    def forward(self, images, kp_map, kp_class):
        """Perform a forward pass

        Args:
            images: The batch of images.
            kp_map: The keypoint map.
            kp_class: The class of the keypoint.

        Returns:
            A dict with the predicted additional error and a sigmoid
            corresponding to if it makes a difference.
        """

        # This is all from the ch-cnn implementation.
        kp_class_one_hot = kp_class.clone()
        conv4 = self.model.conv4(images)
        im_stream = self.model.conv5(conv4)
        im_stream = im_stream.view(im_stream.size(0), -1)
        im_stream = self.model.infer(im_stream)

        # Keypoint Stream
        kp_map = kp_map.view(kp_map.size(0), -1)
        kp_map = self.model.kp_map(kp_map)
        kp_class = self.model.kp_class(kp_class)
        # Concatenate the two keypoint feature vectors
        kp_stream = torch.cat([kp_map, kp_class], dim=1)

        # Softmax followed by reshaping into a 13x13
        # Conv4 as shape batch * 384 * 13 * 13
        kp_stream = F.softmax(self.model.kp_fuse(kp_stream), dim=1)
        kp_stream = kp_stream.view(kp_stream.size(0), 1, 13, 13)

        # Attention -> Elt. wise product, then summation over x and y dims
        kp_stream = kp_stream * conv4
        kp_stream = kp_stream.sum(3).sum(2)

        # Concatenate fc7 and attended features
        fused_embed = torch.cat([im_stream, kp_stream], dim=1)
        fused_embed = self.model.fusion(fused_embed)

        # Go from the CHCNN embedding to the predicted AE.
        output = self.extra_linear(fused_embed)

        # Multiply the prediction by the kp class, since the output is
        # X bins for every KP class.
        output_bce = output[:, :NUM_KEYPOINTS]
        output_bce = torch.sigmoid((output_bce * kp_class_one_hot).sum(dim=1))

        output_cx = output[:, NUM_KEYPOINTS:]
        output_cx = output_cx.view(
            output_cx.shape[0], kp_class_one_hot.shape[1], -1)
        output_cx = (
            output_cx*kp_class_one_hot.unsqueeze(2).repeat(
                1, 1, output_cx.shape[2])).sum(dim=1)

        return {'output_cx': output_cx, 'output_bce': output_bce}

    def get_parameters(self, finetune=False):
        """ Gets the parameters for training.

        Args:
            finetune: a bool telling us whether to train the whole model or
            just the final linear layers.

        Returns:
            the appropriate model parameters.
        """
        if finetune:
            return self.extra_linear.parameters()

        return self.parameters()

    # pylint: disable=R0201
    def get_target(self, additional_error, num_bins):
        """Returns the softmax target based on the additional error and # bins

        Args:
            additional_error: The additional error.
            num_bins: How many bins there are.

        Returns:
            The long target for cx loss.
        """
        # Turn the additional error into an int, and place zero at the center
        # of the softmax output (so it can predict positive and negative
        # additional error.)
        target = torch.floor(additional_error) + num_bins//2

        # If the additional error is greater than the top bin, put it in the
        # top bin.
        target = (target >= num_bins).float()*(num_bins-1)+\
                (1-(target >= num_bins).float())*target

        # If it's below the bottom bin, set it to zero.
        target = (1-(target < 0).float())*target

        # Convert the target to a long.
        target = target.long()

        return target

    def loss(self, network_out, additional_error, use_soft_target):
        """Calculate the loss

        Args:
            network_out: The output of the network.
            additional_error: The actual additional error (the target).

        Returns:
            Cross-entropy error for every sample in the batch.
        """

        num_bins = network_out['output_cx'].shape[1]
        target = self.get_target(additional_error, num_bins)
        # Calculate cross entropy loss
        softmaxed_cx_output = torch.softmax(network_out['output_cx'], dim=1)

        # In order to enable the soft target, we construct a target tensor.
        soft_target = torch.zeros(target.shape[0], num_bins).cuda()
        soft_target[torch.arange(target.shape[0]), target] = 1

        # And if soft target is enabled, we convolve the target with a
        # Gaussian filter.
        if use_soft_target:
            soft_filter = 1/(np.sqrt(2*np.pi)*3)*np.exp(
                -torch.arange(-5, 6).pow(2).float()/(2*9)).unsqueeze(0).cuda()

            soft_target = torch.nn.functional.conv1d(soft_target.unsqueeze(1),
                                                     soft_filter.unsqueeze(1),
                                                     padding=5).squeeze()

        # Put a lower clamp on the softmaxed value, so that we don't get nan
        # loss.
        softmaxed_cx_output = torch.clamp(softmaxed_cx_output, 1e-43, 1.)
        cx_loss = -(soft_target*torch.log(softmaxed_cx_output)).sum(dim=1)

        # Calculate "does it make a difference" loss
        target_bce = 1-(additional_error == 0).float().cuda()
        bce_loss = self.binary_loss_fn(network_out['output_bce'], target_bce)

        return {'bce_loss': bce_loss, 'cx_loss': cx_loss}
