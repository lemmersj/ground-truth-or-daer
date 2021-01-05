"""A dataset for PASCAL3D+"""
import copy
import random
import os

import torch
import torchvision
import pandas
import numpy as np
from PIL import Image

# pylint: disable=invalid-name,too-many-instance-attributes
# pylint: disable=consider-using-enumerate,no-self-use, bare-except

class pascal_dataset(torch.utils.data.Dataset):

    """
        Construct a Pascal Dataset.
        Inputs:
            csv_path    path containing instance data
            augment     boolean for flipping images
    """
    def __init__(self, csv_path, dataset_root=None,
                 im_size=227, transform=None, map_size=46,
                 load_adv=False, load_rand=False, augment=False):

        assert transform is not None

        # Load instance data from csv-file
        (im_paths, bbox, kp_loc, kp_cls,
         obj_cls, vp_labels, targets) = self.csv_to_instances(csv_path)
        targets = targets.squeeze()
        sample_dict = {}

        for i in range(len(im_paths)):
            uid = im_paths[i]+"_"+str(bbox[i][0])+"_"+\
                    str(bbox[i][1])+"_"+str(bbox[i][2])+"_"+\
                    str(bbox[i][3])+"_"+str(kp_cls[i])
            if uid not in sample_dict.keys():
                sample_dict[uid] = {}
            if targets[i] == 0:
                sample_dict[uid]['other_kp'] = kp_loc[i]
            else:
                sample_dict[uid]['bbox'] = bbox[i]
                sample_dict[uid]['im_path'] = im_paths[i]
                sample_dict[uid]['gt_kp'] = kp_loc[i]
                sample_dict[uid]['kp_cls'] = kp_cls[i]
                sample_dict[uid]['obj_cls'] = obj_cls[i]
                sample_dict[uid]['vp_labels'] = vp_labels[i]

                # Setting rand to true ignores the adversarial kp.
                sample_dict[uid]['rand'] = False
                sample_dict[uid]['other_kp'] = [0, 0]

        self.sample_list = []

        # Each row in the sample dict has a true and an adversarial example
        print("Started with " + str(len(sample_dict.keys())) + " samples")
        for key in sample_dict:
            # If you want adversaries, load the whole row as is
            if load_adv:
                self.sample_list.append(dict(sample_dict[key]))
            # if you want random values, set the rand flag.
            # random values are generated in __getitem__
            if load_rand:
                sample_dict[key]['rand'] = True
                self.sample_list.append(dict(sample_dict[key]))
        print("Ended with " + str(len(self.sample_list)) + " keys")
        # dataset parameters
        self.root = dataset_root
        self.loader = self.pil_loader
        self.im_paths = im_paths
        self.bbox = bbox
        self.kp_loc = kp_loc
        self.kp_cls = kp_cls
        self.obj_cls = obj_cls
        self.vp_labels = vp_labels
        self.img_size = im_size
        self.map_size = map_size
        self.transform = transform
        self.augment = augment

    def __getitem__(self, index):
        this_item = self.sample_list[index]
        bbox = list(this_item['bbox'])
        gt_kp = list(this_item['gt_kp'])
        other_kp = list(this_item['other_kp'])
        kp_cls = this_item['kp_cls']
        obj_cls = this_item['obj_cls']
        view = this_item['vp_labels']
        im_path = this_item['im_path']

        # Transform labels
        azim, elev, tilt = (view + 360.) % 360.
        img, kp_loc_gt = self.loader(
            os.path.join(self.root, im_path), bbox, gt_kp)

        kp_rel_crop = copy.deepcopy(kp_loc_gt)

        # if it's not a random sample, use the adversarial example
        if not this_item['rand']:
            img, kp_loc_other = self.loader(
                os.path.join(self.root, im_path), bbox, other_kp)
        else:
            # Otherwise, use a random location
            kp_loc_other = [random.random(), random.random()]

        other_rel_crop = copy.deepcopy(kp_loc_other)
        # Load and transform image
        img = self.transform(img)

        # Generate keypoint map image, and kp class vector
        kpc_vec = torch.zeros((34))
        kpc_vec[int(kp_cls)] = 1
        kp_class = kpc_vec #torch.from_numpy(kpc_vec)

        kp_map_gt = self.generate_kp_map_chebyshev(kp_loc_gt)

        kp_map_other = self.generate_kp_map_chebyshev(kp_loc_other)

        # construct unique key for statistics
        _bb = str(int(bbox[0])) + '-' + str(int(bbox[1])) +\
                '-' + str(int(bbox[2])) + '-' + str(int(bbox[3]))
        key_uid = im_path + '_'  + _bb + '_objc' + str(int(obj_cls)) +\
                '_kpc' + str(int(kp_cls)) + '_rand_'+str(this_item['rand'])

        return (img, azim, elev, tilt, obj_cls, kp_map_gt,
                kp_map_other, kp_class, key_uid, kp_rel_crop,
                other_rel_crop, bbox, index)

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.sample_list)

    def pil_loader(self, path, bbox, kp_loc):
        """Loads an image

        Args:
           path: Absolute image path.
           bbox: 4 element tuple (x_min, y_min, x_max, y_max).
           flip: boolean for flipping image horizontally.
           kp_loc: 2-element tuple (x_loc, y_loc)
        Returns:
            A tuple containing the cropped PIL image and the
            keypoint location as a proportion of the crop.
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

                bbox = copy.deepcopy(bbox)
                if self.augment:
                    # random crop
                    if torch.rand(1) > 0.5:
                        bbox_width = bbox[2] - bbox[0]
                        bbox_height = bbox[3] - bbox[1]
                        min_bbox_left = bbox[0] - 0.2 * bbox_width
                        if min_bbox_left < 0:
                            min_bbox_left = 0
                        max_bbox_left = bbox[0] + 0.2 * bbox_width
                        if max_bbox_left > kp_loc[0]:
                            max_bbox_left = kp_loc[0]

                        min_bbox_right = bbox[2] - 0.2 * bbox_width
                        if min_bbox_right < kp_loc[0]:
                            min_bbox_right = kp_loc[0]
                        max_bbox_right = bbox[2] + 0.2 * bbox_width
                        if max_bbox_right > img.size[0]:
                            max_bbox_right = img.size[0]
                        min_bbox_top = bbox[1] - 0.2 * bbox_height
                        if min_bbox_top < 0:
                            min_bbox_top = 0
                        max_bbox_top = bbox[1] + 0.2 * bbox_height
                        if max_bbox_top > kp_loc[1]:
                            max_bbox_top = kp_loc[1]

                        min_bbox_bottom = bbox[3] - 0.2 * bbox_height
                        if min_bbox_bottom < kp_loc[1]:
                            min_bbox_bottom = kp_loc[1]

                        max_bbox_bottom = bbox[3] + 0.2 * bbox_height
                        if max_bbox_bottom > img.size[1]:
                            max_bbox_bottom = img.size[1]

                        bbox_left = (
                            max_bbox_left - min_bbox_left)*torch.rand(1)+\
                                min_bbox_left
                        bbox_right = (
                            max_bbox_right - min_bbox_right)*torch.rand(1)+\
                                min_bbox_right
                        bbox_top = (
                            max_bbox_top - min_bbox_top)*torch.rand(1)+\
                                min_bbox_top
                        bbox_bottom = (
                            max_bbox_bottom - min_bbox_bottom)*torch.rand(1)+\
                                min_bbox_bottom

                        bbox = [bbox_left.item(), bbox_top.item(),\
                                bbox_right.item(), bbox_bottom.item()]
                    if torch.rand(1) > 0.5:
                        img = img.convert("L")
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

    def unnormalize_image(self, image):
        """Removes normalization from the image for display.

        Args:
            image: The input image as a tensor.
        Returns:
            The unnormalized image as a PIL image.
        """
        toPIL = torchvision.transforms.ToPILImage()
        image[0, :, :] = image[0, :, :] + 104
        image[1, :, :] = image[1, :, :] + 116.668
        image[2, :, :] = image[2, :, :] + 122.678

        image[0, :, :] = image[0, :, :] * 1/255.
        image[1, :, :] = image[1, :, :] * 1/255.
        image[2, :, :] = image[2, :, :] * 1/255.

        image = toPIL(image)
        b, g, r = image.split()
        image = Image.merge("RGB", (r, g, b))

        return image

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

        return (image_paths, bboxes, kp_loc, kp_class,
                obj_class, viewpoints, targets)


    def generate_kp_map_chebyshev(self, kp):
        """ Generate Chebyshev map given a keypoint location

        Args:
            kp: the keypoint location as a proportion of the image crop.

        Returns:
            A 2-dimensional map full of chebyshev distances.
        """
        if kp[0] > 1.:
            kp[0] = 1.
        if kp[0] < 0:
            kp[0] = 0.

        if kp[1] > 1.:
            kp[1] = 1.
        if kp[1] < 0:
            kp[1] = 0.
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
