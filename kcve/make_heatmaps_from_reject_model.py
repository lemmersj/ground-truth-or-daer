"""Makes heatmaps for train or val samples.
"""
import os
import argparse
import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import wandb
from util import Paths, get_data_loaders, metrics
from models import clickhere_cnn, render4cnn
from reject_model import RejectModel
from IPython import embed
from datasets.mturk_dataset import mturk_dataset

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
    #train_loader, valid_loader = get_data_loaders(dataset=args.dataset,
    #                                   batch_size=1,
    #                                   num_workers=16,
    #                                   return_kp=True)
    loader = torch.utils.data.DataLoader(
        dataset=mturk_dataset(
            args.val_csv, args.mturk_csv),
        batch_size=1, shuffle=False, num_workers=16)
    '''if args.split == "train":
        loader = train_loader
    if args.split == "val":
        loader = val_loader'''
    # create the task model.
    model = RejectModel(200)
    model.load_state_dict(torch.load(args.weights))
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

            #heatmap_large[x_coord, y_coord] = heatmap[x_scaled_rounded, y_scaled_rounded]
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
    for i, (images, azim_label, elev_label, tilt_label, kp_class,\
            obj_cls, kp_map_gt, kp_map_turk, _, img_path, gt_kp, turk_kp_loc) in enumerate(data_loader):
        print(i/len(data_loader))
        #wandb.log({"hist_gen_pct":float(i)/len(data_loader)})
        # In my use case, having gt is probably fine, even though I think it
        # isn't technically correct.
        if i < args.start_idx or i > args.end_idx:
            continue
        # relgeodesic is the last one to be created
        '''if os.path.exists(os.path.join(
            args.dir, key_uid[0].split("/")[-1]+".relgeodesicheatmap")):
            print(f"{i} exists")
            continue'''
        # start_idx tells us which kp map we're starting with.
        start_idx = 0
        heatmap_identity = torch.zeros((maps.shape[1], maps.shape[2]))
        with torch.no_grad():
            while True:
                list_len = min(batch_size, maps.shape[0]-start_idx)
                images_repeated = images.repeat(list_len, 1, 1, 1)
                these_maps = maps[start_idx:start_idx+list_len, :, :]
                score = model(images_repeated.cuda(),
                                         these_maps.cuda(),
                                         kp_class.repeat(list_len, 1).float().cuda())
                cx_softmaxed = score['output_cx'].softmax(dim=1)
                values = torch.arange(cx_softmaxed[0, :].shape[0]).cuda().float()
                ae_pred = (values*cx_softmaxed).sum(dim=1) * score['output_bce']
                batch_idx = 0
                for pair_idx in range(start_idx, start_idx+list_len):
                    heatmap_identity[pairs[pair_idx][0].item(),\
                                     pairs[pair_idx][1].item()] =\
                            ae_pred[batch_idx]
                    batch_idx += 1
                start_idx += batch_size
                # If start is past the end, move to the next sample.
                if start_idx >= maps.shape[0]:
                    break
        heatmap_identity = upsample(heatmap_identity, images.shape[2])/200

        image_name = img_path[0].split("/")[-1] 
        filename = f"{image_name}-{kp_class.argmax().item()}-{azim_label.item()}-{elev_label.item()}-{tilt_label.item()}"
        torch.save(
            heatmap_identity, os.path.join(
                args.dir, filename+".identityheatmap"))
        human_image = transform_image(images.squeeze())
        kp_227 = np.floor(np.array([gt_kp[0].item() * 227, gt_kp[1].item() * 227])).astype(int)
        plt.ioff()
        _, ax1 = plt.subplots(1, 1)
        ax1.imshow(human_image)
        ax1.add_artist(plt.Circle((kp_227[0].item(), kp_227[1].item()), 5, color='green', ec='black'))
        heatmap_plot = ax1.imshow(heatmap_identity.transpose(0, 1), cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"]))
        heatmap_plot.set_alpha(0.8)
        plt.title("Predicted Additional Error")
        plt.axis('off')
        plt.savefig(os.path.join(args.dir, filename+"-predicted.pdf"), bbox_inches="tight")
        plt.close()

if __name__ == '__main__':
    plt.switch_backend('Agg')
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--dir', type=str, required=True)
    PARSER.add_argument('--val_csv', type=str, required=True)
    PARSER.add_argument('--mturk_csv', type=str, required=True)
    PARSER.add_argument('--weights', type=str, required=True)
    PARSER.add_argument('--start_idx', type=int, default=0)
    PARSER.add_argument('--end_idx', type=int, default=1e6)

    CMD_ARGS = PARSER.parse_args()
    #wandb.init(project="heatmap_gen_status", config=CMD_ARGS.__dict__)
    main(CMD_ARGS)
