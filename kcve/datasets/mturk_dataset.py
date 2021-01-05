"""A dataset that that loads the mechanical turk annotations.
"""
import csv
import copy
import random
import numpy    as np
import torch
from torchvision    import transforms
from PIL            import Image

#pylint: disable=C0103,W0702,W0123,C0201,R0912,R0201,C0200
class mturk_dataset(torch.utils.data.Dataset):
    """Dataset for parsing a set of mechanical turk keypoints.
    """
    def __init__(self, gs_csv, mturk_csv, random_other=False):
        """Initializes the mturk dataset.

        This initialization includes loading the mturk csv and matching
        mturk annotations to their corresponding gold-standard. Some filtering
        is applied---e.g., the mturk keypoint must be within the same bbox as
        the corresponding gold-standard. Filtering may also be applied such
        that only primary inputs with significant potential angle changes
        are included.

        Args:
            gs_csv: The CSV file containing gold standard annotations.
            mturk_csv: the CSV file containing the mturk KP clicks.
        """
        # Transform is consistent with what is expected by CH-CNN
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(
                mean=(0., 0., 0.), std=(1./255., 1./255., 1./255.)),
             transforms.Normalize(
                 mean=(104, 116.668, 122.678), std=(1., 1., 1.))])

        self.random_other = random_other
        # As are the sizes
        self.img_size = 227
        self.map_size = 46

        # and the keypoint dict
        keypoint_dict = {'bus_body_back_left_lower':0,
                         'bus_body_back_left_upper':1,
                         'bus_body_back_right_lower':2,
                         'bus_body_back_right_upper':3,
                         'bus_body_front_left_upper':4,
                         'bus_body_front_right_upper':5,
                         'bus_body_front_left_lower':6,
                         'bus_body_front_right_lower':7,
                         'bus_left_back_wheel':8,
                         'bus_left_front_wheel':9,
                         'bus_right_back_wheel':10,
                         'bus_right_front_wheel':11,
                         'car_left_front_wheel':12,
                         'car_left_back_wheel':13,
                         'car_right_front_wheel':14,
                         'car_right_back_wheel':15,
                         'car_upper_left_windshield':16,
                         'car_upper_right_windshield':17,
                         'car_upper_left_rearwindow':18,
                         'car_upper_right_rearwindow':19,
                         'car_left_front_light':20,
                         'car_right_front_light':21,
                         'car_left_back_trunk':22,
                         'car_right_back_trunk':23,
                         'motorbike_back_seat':24,
                         'motorbike_front_seat':25,
                         'motorbike_head_center':26,
                         'motorbike_headlight_center':27,
                         'motorbike_left_back_wheel':28,
                         'motorbike_left_front_wheel':29,
                         'motorbike_left_handle_center':30,
                         'motorbike_right_back_wheel':31,
                         'motorbike_right_front_wheel':32,
                         'motorbike_right_handle_center':33
                        }

        # We loop through every row in the ground-truth CSV and find the
        # matching mturk click. We save it in a dict (instead of a list)
        # so we can match the two. The key is the image and keypoint class.
        image_dict = {}
        with open(gs_csv) as infile:
            annotations_dictreader = csv.DictReader(infile)
            for row in annotations_dictreader:
                # construct the uid from the image and keypoint
                row_uid = row['imgPath']+"_"+row['keyptClass']

                # a uid may have more than one associated row.
                # I guess this means it's not really a UID...
                if row_uid not in image_dict:
                    image_dict[row_uid] = []

                # Build a dict for this row.
                this_instance = {}
                for key in row:
                    try:
                        this_instance[key] = eval(row[key])
                    except:
                        this_instance[key] = row[key]
                # and save it.
                image_dict[row_uid].append(this_instance)

        # Now we want to build and equivalent dictionary for the
        # mturk annotations.
        annotations_dict = {}
        with open(mturk_csv) as infile:
            annotations_dictreader = csv.DictReader(infile)

            num_annotations = 0
            for row in annotations_dictreader:
                num_annotations += 1
                # Nothing fancy here, just manually chopping to a common
                # start directory. Will need to change if input CSV
                # changes.
                image_loc = row['Input.image_url'][6:]
                kp_class_int = keypoint_dict[row['Input.object_name'].split(
                    "(")[0]+"_"+row['Input.object']]
                row_uid = image_loc+"_"+str(kp_class_int)
                if row_uid not in annotations_dict.keys():
                    annotations_dict[row_uid] = []

                # One row may have many keypoints, since they are
                # evaluated per-image.
                for mturk_sample in eval(
                        row['Answer.annotatedResult.keypoints']):
                    annotations_dict[row_uid].append(mturk_sample)

        # now we do the matching!
        matched_samples = []
        # loop through every image
        for key in image_dict.keys():
            # if there's no matching annotation (uid), skip.
            try:
                candidates = annotations_dict[key]
            except:
                continue

            # For every ground truth with a uid, we choose the best kp.
            for gt_sample in image_dict[key]:
                min_dist = 10000
                min_sample = -1

                # loop through all the mturk samples for this image/kp
                # pair.
                for mturk_sample in candidates:
                    # pick the closest one.
                    dist = np.sqrt(
                        pow(mturk_sample['x'] - \
                            gt_sample['imgKeyptX'], 2) + pow(
                                mturk_sample['y'] - \
                                gt_sample['imgKeyptY'], 2))

                    dist_from_tlx = mturk_sample['x'] - gt_sample['bboxTLX']
                    dist_from_tly = mturk_sample['y'] - gt_sample['bboxTLY']
                    dist_from_brx = mturk_sample['x'] - gt_sample['bboxBRX']
                    dist_from_bry = mturk_sample['y'] - gt_sample['bboxBRY']

                    # reject anything that isn't within 10% of the bbox
                    # width.
                    if dist_from_tlx < 0:
                        continue
                    if dist_from_tly < 0:
                        continue
                    if dist_from_brx > 0:
                        continue
                    if dist_from_bry > 0:
                        continue

                    if dist < min_dist:
                        min_dist = dist
                        min_sample = mturk_sample

                if min_dist < 10000:
                    gt_sample['mturk_x'] = min_sample['x']
                    gt_sample['mturk_y'] = min_sample['y']
                    matched_samples.append(gt_sample)


        self.samples = matched_samples

        print(str(num_annotations) + " annotations.")
        print(str(len(matched_samples)) + " matched gold-standard keys.")

    def __getitem__(self, index):
        """Gets an item at a specified index.

        args:
            index: The index at which we are getting the item.

        returns:
            A whole lot of stuff, including images, kp maps, and the target.
        """

        matched_sample = self.samples[index]

        # not optimal to do this load twice, but fine for eval
        image, gt_kp_loc = self.pil_loader(
            "/z/dat/PASCAL3D/"+matched_sample['imgPath'],\
                          [matched_sample['bboxTLX'],\
                          matched_sample['bboxTLY'],\
                          matched_sample['bboxBRX'],\
                          matched_sample['bboxBRY']],\
                          [matched_sample['imgKeyptX'],\
                          matched_sample['imgKeyptY']])
        image, turk_kp_loc = self.pil_loader(
            "/z/dat/PASCAL3D/"+matched_sample['imgPath'],\
                          [matched_sample['bboxTLX'],\
                          matched_sample['bboxTLY'],\
                          matched_sample['bboxBRX'],\
                          matched_sample['bboxBRY']],\
                          [matched_sample['mturk_x'],\
                          matched_sample['mturk_y']])

        if self.random_other:
            turk_kp_loc = [random.random(), random.random()]

        for i in range(len(turk_kp_loc)):
            if turk_kp_loc[i] < 0:
                turk_kp_loc[i] = 0
            if turk_kp_loc[i] > 1:
                turk_kp_loc[i] = 1

        keypoint_dist = np.sqrt(
            pow(matched_sample['imgKeyptX'] - matched_sample['mturk_x'], 2)+\
            pow(matched_sample['imgKeyptY'] - matched_sample['mturk_y'], 2))

        azim = (matched_sample['azimuthClass'] + 360) % 360
        elev = (matched_sample['elevationClass'] + 360) % 360
        tilt = (matched_sample['rotationClass'] + 360) % 360

        image = self.transform(image)
        kpc_vec = np.zeros((34))
        kpc_vec[matched_sample['keyptClass']] = 1
        kpm_map_gt = self.generate_kp_map_chebyshev(
            copy.deepcopy(gt_kp_loc))
        kp_map_gt = torch.from_numpy(kpm_map_gt).float()

        kpm_map_turk = self.generate_kp_map_chebyshev(
            copy.deepcopy(turk_kp_loc))
        kp_map_turk = torch.from_numpy(kpm_map_turk).float()

        return image, azim, elev, tilt, kpc_vec, matched_sample['objClass'],\
                kp_map_gt, kp_map_turk, keypoint_dist,\
                matched_sample['imgPath'], gt_kp_loc, turk_kp_loc

    def __len__(self):
        """Get the dataset length

        Args:
            None
        Returns:
            # of elements in the dataset.
        """
        return len(self.samples)

    def pil_loader(self, path, bbox, kp_loc):
        """Loads an image.

        This includes cropping and some color manipulation (but not
        normalizing)

        Args:
            path: absolute image path.
            bbox: (x_min, y_min, x_max, y_max)
            kp_loc: (x_loc, y_loc)

        Returns:
            A loaded, modified image.
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

    def reverse_transform(self, input_tensor):
        """Converts a crop back to the color space for display.

        Args:
            input_tensor: a torch tensor normalized for CHCNN.

        Returns:
            A channel-flipped, unnormalized tensor.
        """
        input_tensor[0, :, :] = input_tensor[0, :, :] + 104
        input_tensor[1, :, :] = input_tensor[1, :, :] + 116.668
        input_tensor[2, :, :] = input_tensor[2, :, :] + 122.678

        input_tensor[0, :, :] = input_tensor[0, :, :] / 255
        input_tensor[1, :, :] = input_tensor[1, :, :] / 255
        input_tensor[2, :, :] = input_tensor[2, :, :] / 255

        new_input_tensor = input_tensor.clone()
        new_input_tensor[0, :, :] = input_tensor[2, :, :]
        new_input_tensor[2, :, :] = input_tensor[0, :, :]

        return new_input_tensor

    def generate_kp_map_chebyshev(self, kp, map_size=46):
        """Generates a chebyshev keypoint map.

        Args:
            kp: the keypoint as a tuple, in units of proportion of image.
            map_size: How big the keypoint map should be.

        Returns:
            A keypoint map, in units of proportion, of size map_size,
            consisting of the distance of every pixel from the clicked one.
        """


        assert kp[0] >= 0. and kp[0] <= 1., kp
        assert kp[1] >= 0. and kp[1] <= 1., kp
        kp_map = np.ndarray((map_size, map_size))


        kp[0] = kp[0] * map_size
        kp[1] = kp[1] * map_size

        for i in range(0, map_size):
            for j in range(0, map_size):
                kp_map[i, j] = max(np.abs(i - kp[0]), np.abs(j - kp[1]))

        # Normalize by dividing by the maximum possible value,
        # which is self.IMG_SIZE -1
        kp_map = kp_map / (1. * map_size)

        return kp_map
