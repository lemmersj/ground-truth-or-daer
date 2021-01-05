"""Dataset to load both real and synthetic data.
"""
import copy
import random
import os

import torch
import pandas
import numpy    as np
from PIL            import Image
import torchvision

#pylint: disable=C0103,R0902,R0201,W0702

class both_dataset(torch.utils.data.Dataset):

    """
        Construct a Pascal Dataset.
        Inputs:
            csv_path    path containing instance data
            augment     boolean for flipping images
    """
    def __init__(self, synthetic_csv, pascal_csv,
                 synthetic_root, pascal_root,
                 im_size=227, transform=None,
                 map_size=46):

        assert transform is not None

        # Load instance data from csv files

        # synthetic
        (self.im_paths_syn, self.bbox_syn, self.kp_loc_syn,
         self.kp_cls_syn, self.obj_cls_syn, self.vp_labels_syn,
         self.targets_syn) = self.csv_to_instances(synthetic_csv)
        self.targets_syn = self.targets_syn.squeeze()

        # real (PASCAL3D+)
        (self.im_paths_real, self.bbox_real, self.kp_loc_real,
         self.kp_cls_real, self.obj_cls_real, self.vp_labels_real,
         self.targets_real) = self.csv_to_instances(pascal_csv)

        self.targets_real = self.targets_real.squeeze()

        self.synthetic_root = synthetic_root
        self.pascal_root = pascal_root

        self.loader = self.pil_loader
        self.img_size = im_size
        self.transform = transform
        self.map_size = map_size

    def __getitem__(self, index):
        """Returns the object occresponding to the given index.

        Args:
        index: The integer value of the index to return.

        Returns:
        Lots of information related to the corresponding object.
        """

        # Real data has lowest numbers.
        if index >= len(self.im_paths_real):
            index = index - len(self.im_paths_real)
            im_path = self.im_paths_syn[index]
            bbox = list(self.bbox_syn[index])
            gt_kp = list(self.kp_loc_syn[index])
            kp_cls = self.kp_cls_syn[index]
            obj_cls = self.obj_cls_syn[index]
            view = self.vp_labels_syn[index]
            dataset_root = self.synthetic_root
        else:
            im_path = self.im_paths_real[index]
            bbox = list(self.bbox_real[index])
            gt_kp = list(self.kp_loc_real[index])
            kp_cls = self.kp_cls_real[index]
            obj_cls = self.obj_cls_real[index]
            view = self.vp_labels_real[index]

            dataset_root = self.pascal_root

        # Transform labels from -180->180 to 0->360
        azim, elev, tilt = (view + 360.) % 360.

        # Load the image
        img, kp_loc_gt = self.loader(
            os.path.join(dataset_root, im_path), bbox, gt_kp)

        # Save locations relative to the crop.
        kp_rel_crop = copy.deepcopy(kp_loc_gt)
        kp_loc_other = [random.random(), random.random()]
        other_rel_crop = copy.deepcopy(kp_loc_other)

        # Load and transform image
        img = self.transform(img)

        # Generate keypoint map image, and kp class vector
        kpc_vec = torch.zeros((34))
        kpc_vec[int(kp_cls)] = 1
        kp_class = kpc_vec

        # This modifies the keypoint locations, which is why there's a
        # deep copy above.
        kp_map_gt = self.generate_kp_map_chebyshev(kp_loc_gt)
        kp_map_other = self.generate_kp_map_chebyshev(kp_loc_other)

        # construct unique key for statistics
        _bb = str(int(bbox[0])) + '-' + str(int(bbox[1])) + '-' +\
        str(int(bbox[2])) + '-' + str(int(bbox[3]))
        key_uid = im_path + '_'  + _bb +\
                '_objc' + str(int(obj_cls)) + '_kpc' + str(int(kp_cls))

        return (img, azim, elev, tilt, obj_cls, kp_map_gt, kp_map_other,
                kp_class, key_uid, kp_rel_crop, other_rel_crop, bbox, index)

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.im_paths_syn) + len(self.im_paths_real)

    def pil_loader(self, path, bbox, kp_loc):
        """
        Image loader
        Args:
            path: absolute image path
            bbox: 4-element tuple (x_min, y_min, x_max, y_max)
            flip: boolean for flipping image horizontally
            kp_loc: 2-element tuple (x_loc, y_loc)
        """
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                # Calculate relative kp_loc position
                if bbox[0] == -1:
                    bbox[0] = 0
                    bbox[1] = 0
                    bbox[2] = img.size[0]
                    bbox[3] = img.size[1]

                kp_loc[0] = float(kp_loc[0]-bbox[0])/float(bbox[2]-bbox[0])
                kp_loc[1] = float(kp_loc[1]-bbox[1])/float(bbox[3]-bbox[1])

                # Convert to RGB, crop, and resize
                img = img.convert('RGB')

                # Convert to BGR from RGB
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))
                img = img.crop(box=bbox)
                img = img.resize((self.img_size, self.img_size), Image.LANCZOS)

                return img, kp_loc

    def unnormalize_image(self, image_in):
        """Removes image normalization for display.

        Args:
            image_in: The input tensor.

        Returns:
            A pillow image with the normalization removed.
        """
        toPIL = torchvision.transforms.ToPILImage()
        image_in[0, :, :] = image_in[0, :, :] + 104
        image_in[1, :, :] = image_in[1, :, :] + 116.668
        image_in[2, :, :] = image_in[2, :, :] + 122.678

        image_in[0, :, :] = image_in[0, :, :] * 1/255.
        image_in[1, :, :] = image_in[1, :, :] * 1/255.
        image_in[2, :, :] = image_in[2, :, :] * 1/255.

        image_out = toPIL(image_in)
        b, g, r = image_out.split()
        image_out = Image.merge("RGB", (r, g, b))

        return image_out

    def csv_to_instances(self, csv_path):
        """Convert CSV file to instances.

        Args:
            csv_path: The location of the csv file.

        Returns:
            A tuple containing the parsed csv.
        """
        df = pandas.read_csv(csv_path, sep=',')
        data = df.values

        data_split = np.split(data, [0, 1, 5, 7, 8, 9, 12], axis=1)
        del data_split[0]

        image_paths = np.squeeze(data_split[0]).tolist()

        bboxes = data_split[1].tolist()
        kp_loc = data_split[2].tolist()
        kp_class = np.squeeze(data_split[3]).tolist()
        obj_class = np.squeeze(data_split[4]).tolist()
        viewpoints = np.array(data_split[5].tolist())
        try:
            targets = data_split[6]
        except:
            targets = np.ones(len(data_split[6]))

        if targets.shape[1] > 0:
            image_paths = list(
                np.array(image_paths)[np.where(targets == 1)[0]])
            bboxes = list(
                np.array(bboxes)[np.where(targets == 1)[0]])
            kp_loc = list(
                np.array(kp_loc)[np.where(targets == 1)[0]])
            kp_class = list(
                np.array(kp_class)[np.where(targets == 1)[0]])
            obj_class = list(
                np.array(obj_class)[np.where(targets == 1)[0]])

        return (image_paths, bboxes, kp_loc, kp_class,
                obj_class, viewpoints, targets)


    def generate_kp_map_chebyshev(self, kp):
        """ Generate Chebyshev map given a keypoint location

        Args:
            kp: the keypoint location as a proportion of the image crop.

        Returns:
            A 2-dimensional map full of chebyshev distances.
        """
        assert kp[0] >= 0. and kp[0] <= 1., kp
        assert kp[1] >= 0. and kp[1] <= 1., kp
        kp_map = torch.zeros((self.map_size, self.map_size))


        kp[0] = kp[0] * self.map_size
        kp[1] = kp[1] * self.map_size

        for i in range(0, self.map_size):
            for j in range(0, self.map_size):
                kp_map[i, j] = max(np.abs(i - kp[0]), np.abs(j - kp[1]))

        # Normalize by dividing by the maximum possible value,
        # which is self.IMG_SIZE -1
        kp_map = kp_map / (1. * self.map_size)

        return kp_map
