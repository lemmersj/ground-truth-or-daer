"""Click Here-CNN model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

#pylint: disable=R0902,C0103

class clickhere_cnn(nn.Module):
    """The Click Here-cnn model.
    """
    def __init__(self, renderCNN, weights_path=None):
        """Initialize the CHCNN model.

        Args:
            renderCNN: The original R4CNN weights.
            weights_path: The CHCNN weights to load.
            num_classes: The number of classes (car, motorcyclle, bus)
        """
        super(clickhere_cnn, self).__init__()

        # Image Stream
        self.conv4 = renderCNN.conv4
        self.conv5 = renderCNN.conv5

        self.infer = nn.Sequential(
                        nn.Linear(9216, 4096),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(4096, 4096),
                        nn.ReLU(),
                        nn.Dropout(0.5))

        #Keypoint Stream
        self.kp_map = nn.Linear(2116, 2116)
        self.kp_class = nn.Linear(34, 34)
        self.kp_fuse = nn.Linear(2150, 169)
        self.pool_map = nn.MaxPool2d((5, 5), (5, 5), (1, 1), ceil_mode=True)

        # Fused layer
        self.fusion = nn.Sequential(
            nn.Linear(4096 + 384, 4096),
            nn.ReLU(), nn.Dropout(0.5))

        # Prediction layers
        self.azim = nn.Linear(4096, 12*360)
        self.elev = nn.Linear(4096, 12*360)
        self.tilt = nn.Linear(4096, 12*360)

        if weights_path is not None:
            self.init_weights(weights_path)


    def init_weights(self, weights_path):
        """Load the model weights.

        Args:
            weights_path: The path to the weights.

        Returns:
            Nothing.
        """
        self.load_state_dict(torch.load(weights_path))

    def forward(self, images, kp_map, kp_cls, obj_class):
        """Performs a forward pass.

        Args:
            images: a batch of imae tensors.
            kp_map: Chebyshev maps corresponding to the keypoint clicks.
            kp_class: Class of the keypoint.
            obj_class: Class of the object.

        Returns:
            Predicted azimuth, elevation, and tilt.
        """

        obj_class = obj_class.long()
        # Image Stream
        conv4 = self.conv4(images)
        im_stream = self.conv5(conv4)
        im_stream = im_stream.view(im_stream.size(0), -1)
        im_stream = self.infer(im_stream)

        # Keypoint Stream
        kp_map = kp_map.view(kp_map.size(0), -1)
        kp_map = self.kp_map(kp_map)
        kp_cls = self.kp_class(kp_cls)

        # Concatenate the two keypoint feature vectors
        kp_stream = torch.cat([kp_map, kp_cls], dim=1)

        # Softmax followed by reshaping into a 13x13
        # Conv4 as shape batch * 384 * 13 * 13
        kp_stream = F.softmax(self.kp_fuse(kp_stream), dim=1)
        kp_stream = kp_stream.view(kp_stream.size(0), 1, 13, 13)

        # Attention -> Elt. wise product, then summation over x and y dims
        kp_stream = kp_stream * conv4
        kp_stream = kp_stream.sum(3).sum(2)

        # Concatenate fc7 and attended features
        fused_embed = torch.cat([im_stream, kp_stream], dim=1)
        fused_embed = self.fusion(fused_embed)

        # Final inference
        azim = self.azim(fused_embed)
        elev = self.elev(fused_embed)
        tilt = self.tilt(fused_embed)

        # mask on class
        azim = self.azim(fused_embed)
        azim = azim.view(-1, 12, 360)
        azim = azim[torch.arange(fused_embed.shape[0]), obj_class, :]
        elev = self.elev(fused_embed)
        elev = elev.view(-1, 12, 360)
        elev = elev[torch.arange(fused_embed.shape[0]), obj_class, :]
        tilt = self.tilt(fused_embed)
        tilt = tilt.view(-1, 12, 360)
        tilt = tilt[torch.arange(fused_embed.shape[0]), obj_class, :]

        return azim, tilt, elev
