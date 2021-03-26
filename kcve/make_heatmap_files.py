"""Makes heatmaps for train or val samples.
"""
import os
import argparse
import torch
import torchvision
import numpy as np
from util import Paths, get_data_loaders, metrics
from models import clickhere_cnn, render4cnn

#pylint: disable=R0914

def main(args):
    """Setup for heatmap generation.

    Makes the output directory, loads the dataset model, and executes the
    function that produces the heatmaps.

    Args:
        args: Command line arguments.
    """

    # Make the output directory. Or don't, if it has already been made.
    try:
        os.makedirs(args.dir)
    except OSError:
        pass

    # Load the dataset. Can switch between train or val loaders here.
    train_loader, valid_loader = get_data_loaders(dataset=args.dataset,
                                                  batch_size=1,
                                                  num_workers=16,
                                                  return_kp=True)
    if args.split == "train":
        loader = train_loader
    if args.split == "val":
        loader = valid_loader
    # create the task model.
    model = clickhere_cnn(render4cnn(), weights_path=Paths.clickhere_weights)
    # train/evaluate on GPU
    model.cuda()

    # Generate the heatmap.
    eval_step(model=model,
              data_loader=loader,
              args=args)

def generate_kp_map_chebyshev(keypoints, map_size):
    """Generates a chebyshev map for a given keypoint.

    Args:
        kps: The keypoint in terms of proportion of image. An (X,Y) tuple.
        map_size: The size of the map to return in pixels.

    Returns:
        A normalized map where every cell represents the chebyshev distance.
    """
    # If the keypoint is outside the image, something is wrong.
    assert (keypoints >= 0.).all() and (keypoints < 1.).all(), keypoints

    # Bring the image up to the desired map.
    keypoints = keypoints * map_size

    # Tensor magic.
    x_group = torch.arange(map_size).float().unsqueeze(0).repeat(
        keypoints.shape[0], 1)
    y_group = torch.arange(map_size).float().unsqueeze(0).repeat(
        keypoints.shape[0], 1)
    x_dist = torch.abs(
        keypoints[:, 0].unsqueeze(1).repeat(1, map_size) - x_group)\
            .unsqueeze(2).repeat(1, 1, map_size)
    y_dist = torch.abs(
        keypoints[:, 1].unsqueeze(1).repeat(1, map_size) - y_group)\
            .unsqueeze(1).repeat(1, map_size, 1)
    dist = torch.cat((x_dist.unsqueeze(0), y_dist.unsqueeze(0)), dim=0)
    kp_map = torch.max(dist, dim=0)[0]

    # Normalize by dividing by the maximum possible value, which is
    # self.IMG_SIZE -1
    kp_map = kp_map / (1. * map_size)

    return kp_map

def transform_image(image):
    """Reverses the normalization used by CHCNN.

    Args:
        image: The image.
    Returns:
        A version of the image for display.
    """
    to_pil = torchvision.transforms.ToPILImage()
    image[0, :, :] += 104
    image[1, :, :] += 116.668
    image[2, :, :] += 122.678

    image = image/255

    image_2 = torch.zeros(image.shape)
    image_2[0, :, :] = image[2, :, :]
    image_2[1, :, :] = image[1, :, :]
    image_2[2, :, :] = image[0, :, :]

    return to_pil(image_2)

def upsample(heatmap, out_dim):
    """Upsamples the heatmap for display purposes.

    Args:
        heatmap: The 46x46 output heatmap.
        out_dim: How big the output image is.
    Returns:
        Upsampled version of the heatmap.
    """
    # extend the heatmap in order to make interpolation easier
    heatmap_extended = torch.zeros((heatmap.shape[0]+1, heatmap.shape[1]+1))
    heatmap_extended[1:, :heatmap.shape[1]] = heatmap.clone()
    heatmap_extended[:heatmap.shape[0], 1:] = heatmap.clone()
    heatmap_extended[:heatmap.shape[0], :heatmap.shape[1]] = heatmap.clone()
    heatmap_extended[heatmap.shape[0], heatmap.shape[1]] = \
            heatmap[heatmap.shape[0]-1, heatmap.shape[1]-1].clone()
    heatmap_large = torch.zeros((out_dim, out_dim))
    for x_coord in range(out_dim):
        for y_coord in range(out_dim):
            x_scaled = (x_coord/out_dim)*46
            x_scaled_floored = int(np.floor(x_scaled))
            x_scaled_roofed = x_scaled_floored+1
            #x_scaled_rounded = int(np.round(x_scaled))
            y_scaled = (y_coord/out_dim)*46
            #y_scaled_rounded = int(np.round(y_scaled))

            y_scaled_floored = int(np.floor(y_scaled))
            y_scaled_roofed = y_scaled_floored+1
            # bilinear interpolation

            f_x_y1 = (x_scaled_roofed - x_scaled)*heatmap_extended[
                x_scaled_floored, y_scaled_floored] + (
                    x_scaled - x_scaled_floored) * heatmap_extended[
                        x_scaled_roofed, y_scaled_floored]
            f_x_y2 = (x_scaled_roofed - x_scaled)*heatmap_extended[
                x_scaled_floored, y_scaled_roofed] + (
                    x_scaled - x_scaled_floored) * heatmap_extended[
                        x_scaled_roofed, y_scaled_roofed]
            heatmap_large[x_coord, y_coord] = (
                y_scaled_roofed - y_scaled) * f_x_y1 + (
                    y_scaled-y_scaled_floored)*f_x_y2

    return heatmap_large

def eval_step(model, data_loader, args):
    """Generates all the heatmaps.

    Args:
        model: The task model.
        data_loader: The dataloader.
    Returns:
        Nothing. Images are saved to the output directory.
    """
    model.eval()
    pairs = torch.zeros(0, 2)
    # Generate all possible XY pairs
    for x_loc in range(46):
        for y_loc in range(46):
            with torch.no_grad():
                pairs = torch.cat((pairs, torch.tensor(
                    (x_loc, y_loc)).view(1, 2).float()), dim=0)

    # And then generate all possible chebyshev maps.
    maps = generate_kp_map_chebyshev(pairs.clone()/46., 46)

    # cast all possible pairs as an int, so we can use them as indices.
    pairs = pairs.int()

    # Loop through whole dataset.
    batch_size = 256
    for i, (images, azim_label, elev_label, tilt_label, obj_class,\
            _, _, kp_class, key_uid,\
            gt_kp, _, _, _) in enumerate(data_loader):
        print(i/len(data_loader))
        #wandb.log({"hist_gen_pct":float(i)/len(data_loader)})
        # In my use case, having gt is probably fine, even though I think it
        # isn't technically correct.
        if i < args.start_idx or i > args.end_idx:
            continue
        # relgeodesic is the last one to be created
        if os.path.exists(os.path.join(
                args.dir, key_uid[0].split("/")[-1]+".relgeodesicheatmap")):
            print(f"{i} exists")
            continue
        # start_idx tells us which kp map we're starting with.
        start_idx = 0
        heatmap_geodesic = torch.zeros((maps.shape[1], maps.shape[2]))
        heatmap_identity = torch.zeros((maps.shape[1], maps.shape[2]))
        with torch.no_grad():
            while True:
                list_len = min(batch_size, maps.shape[0]-start_idx)
                images_repeated = images.repeat(list_len, 1, 1, 1)
                these_maps = maps[start_idx:start_idx+list_len, :, :]
                azim, elev, tilt = model(images_repeated.cuda(),
                                         these_maps.cuda(),
                                         kp_class.repeat(list_len, 1).cuda(),
                                         obj_class.repeat(list_len).cuda())
                batch_idx = 0
                for pair_idx in range(start_idx, start_idx+list_len):
                    heatmap_geodesic[pairs[pair_idx][0].item(),\
                                     pairs[pair_idx][1].item()] =\
                            metrics.compute_angle_dists([
                                azim[batch_idx, :].argmax().float()\
                                .unsqueeze(0), elev[batch_idx, :].argmax()\
                                .float().unsqueeze(0), tilt[batch_idx, :]\
                                .argmax().float().unsqueeze(0)],\
                                [azim_label, elev_label,\
                                 tilt_label])
                    heatmap_identity[pairs[pair_idx][0].item(),\
                                     pairs[pair_idx][1].item()] =\
                            metrics.compute_dist_from_eye([
                                azim[batch_idx, :].argmax().float()\
                                .unsqueeze(0), elev[batch_idx, :].argmax()\
                                .float().unsqueeze(0), tilt[batch_idx, :]\
                                .argmax().float().unsqueeze(0)],\
                                [azim_label, elev_label,\
                                 tilt_label])
                    batch_idx += 1
                start_idx += batch_size
                # If start is past the end, move to the next sample.
                if start_idx >= maps.shape[0]:
                    break
        kp_46 = np.floor(np.array(
            [gt_kp[0].item() * 46, gt_kp[1].item() * 46])).astype(int)
        heatmap_rel_geodesic = heatmap_geodesic - \
                heatmap_geodesic[kp_46[0], kp_46[1]]
        heatmap_geodesic = upsample(heatmap_geodesic, images.shape[2])
        heatmap_identity = upsample(heatmap_identity, images.shape[2])
        heatmap_rel_geodesic = upsample(heatmap_rel_geodesic, images.shape[2])

        torch.save(
            heatmap_geodesic, os.path.join(
                args.dir, key_uid[0].split("/")[-1]+".geodesicheatmap"))
        torch.save(
            heatmap_identity, os.path.join(
                args.dir, key_uid[0].split("/")[-1]+".identityheatmap"))
        torch.save(
            heatmap_rel_geodesic, os.path.join(
                args.dir, key_uid[0].split("/")[-1]+".relgeodesicheatmap"))

if __name__ == '__main__':
    #plt.switch_backend('Agg')
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--dir', type=str, required=True)
    PARSER.add_argument('--dataset', type=str, required=True)
    PARSER.add_argument('--split', type=str, required=True)
    PARSER.add_argument('--start_idx', type=int, default=0)
    PARSER.add_argument('--end_idx', type=int, default=1e6)

    CMD_ARGS = PARSER.parse_args()
    #wandb.init(project="heatmap_gen_status", config=CMD_ARGS.__dict__)
    main(CMD_ARGS)
