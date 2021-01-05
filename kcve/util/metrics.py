"""Metrics for keypoint-conditioned viewpoint estimation"""
import io
import sys

import numpy as np
from scipy import linalg as linAlg
import torch

#pylint: disable=C0103

def compute_dist_from_eye(preds, labels):
    """Calculate the distance from the identity matrix.

    From Larochelle et al.

    Args:
        preds: Predicted azim, elev, and tilt as a list.
        labels: Gold-standard azim, elev, and tilt as a list.

    Returns:
        The metric of Larochelle et al.
    """
    # Get rotation matrices from prediction and ground truth angles
    predR = angle2dcm(preds[0], preds[1], preds[2])
    gtR = angle2dcm(labels[0], labels[1], labels[2])

    # Get geodesic distance
    errors = torch.zeros(predR.shape[0]).cuda()
    matrix_mult = torch.matmul(predR.transpose(1, 2), gtR)
    errors = torch.norm(torch.eye(3).unsqueeze(0)\
                        .repeat(matrix_mult.shape[0], 1, 1).cuda()-\
                        matrix_mult, dim=(1, 2))
    return errors/np.sqrt(2)

def compute_angle_dists(preds, labels):
    """Calculate the geodesic distance.

    Args:
        preds: Predicted azim, elev, and tilt as a list.
        labels: Gold-standard azim, elev, and tilt as a list.

    Returns:
        The geodesic distance between the two provided angles.
    """
    # Get rotation matrices from prediction and ground truth angles
    predR = angle2dcm(preds[0], preds[1], preds[2])
    gtR = angle2dcm(labels[0], labels[1], labels[2])

    # Get geodesic distance
    errors = torch.zeros(predR.shape[0]).cuda()
    matrix_mult = torch.matmul(predR.transpose(1, 2), gtR)
    text_trap = io.StringIO()
    sys.stdout = text_trap
    matrix_mult = matrix_mult.cpu()
    for i in range(predR.shape[0]):
        # I'm not going to mess with pylint linkages.
        #pylint: disable=E1101
        errors[i] = linAlg.norm((linAlg.logm(matrix_mult[i, :, :], 2)))
    sys.stdout = sys.__stdout__
    return errors/np.sqrt(2)

def angle2dcm(xRot, yRot, zRot, deg_type='deg'):
    """Convert an angle to a rotation matrix.

    Args:
        xRot: Rotation angle around the x axis.
        yRot: Rotation angle around the y axis.
        zRot: Rotation angle around the z axis.
        deg_type: Degrees or rad of previous args.

    Returns:
        A rotation matrix corresponding to the input angles.
    """
    if deg_type == 'deg':
        xRot = xRot * np.pi / 180.0
        yRot = yRot * np.pi / 180.0
        zRot = zRot * np.pi / 180.0
    xMat = torch.eye(3, 3).unsqueeze(0).repeat(xRot.shape[0], 1, 1).cuda()
    xMat[:, 0, 0] = torch.cos(xRot)
    xMat[:, 0, 1] = torch.sin(xRot)
    xMat[:, 1, 0] = -1*torch.sin(xRot)
    xMat[:, 1, 1] = torch.cos(xRot)

    yMat = torch.eye(3, 3).unsqueeze(0).repeat(xRot.shape[0], 1, 1).cuda()
    yMat[:, 0, 0] = torch.cos(yRot)
    yMat[:, 0, 2] = -torch.sin(yRot)
    yMat[:, 2, 0] = torch.sin(yRot)
    yMat[:, 2, 2] = torch.cos(yRot)

    zMat = torch.eye(3, 3).unsqueeze(0).repeat(xRot.shape[0], 1, 1).cuda()
    zMat[:, 1, 1] = torch.cos(zRot)
    zMat[:, 1, 2] = torch.sin(zRot)
    zMat[:, 2, 1] = -torch.sin(zRot)
    zMat[:, 2, 2] = torch.cos(zRot)

    return torch.matmul(zMat, torch.matmul(yMat, xMat))
