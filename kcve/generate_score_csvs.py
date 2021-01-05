"""Generate a score csv.

Called from run_analysis.py, accepts arguments including weights and
scoring method, and outputs a CSV file containing scores and corresponding
errors which can be used by calc_additional_risk.py.
"""
import argparse
import numpy as np
import torch
from torch import nn
import wandb
from util import SoftmaxVPLoss, Paths
from util.metrics import compute_angle_dists
from models import clickhere_cnn, render4cnn
from datasets.mturk_dataset import mturk_dataset
from reject_model import RejectModel

# pylint: disable=too-many-locals, consider-using-enumerate
# pylint: disable=too-many-boolean-expressions, too-many-branches

def entropy(value):
    """Score based on entropy of the task model output distribution

    Args:
        value: The output of the task model
    Returns:
        The entropy of the task model output.
    """
    softmaxed = nn.functional.softmax(value[0], dim=1)

    val_entropy = -1*(softmaxed * torch.log(softmaxed)).sum(dim=1)

    return val_entropy

def maxed_softmax(value):
    """Score based on softmax response of the task model output distribution

    Args:
        value: The output of the task model
    Returns:
        The multiplicative inverse of the maximum value in the softmax.
    """
    softmaxed = nn.functional.softmax(value[0], dim=1)

    return -1*softmaxed.max()

def sampler(value, percentile):
    """Score based on sampling task model output distribution

    Args:
        value: The output of the task model
        percentile: the (sorted) index of the sample we use
    Returns:
        The percentile largest distance from the mean of the samples.
    """
    softmaxed = nn.functional.softmax(value[0], dim=1)
    samples = torch.tensor(
        np.array(
            list(
                torch.utils.data.WeightedRandomSampler(
                    softmaxed, 10000)))).float()
    mean_value = samples.mean(dim=1)
    dist_from_mean = torch.abs(((
        samples-mean_value.unsqueeze(1).repeat(
            1, samples.shape[1]))+180)%360 - 180)
    sorted_val = torch.sort(dist_from_mean).values
    if percentile == 10000:
        percentile = percentile-1
    return sorted_val[:, percentile]

def sampler_binned(value, percentile):
    """Score based on sampling rejection model output distribution

    Args:
        value: The output of the rejection model
        percentile: the (sorted) index of the sample we use
    Returns:
        The percentile largest predicted additional error, with
        mean secondary scoring.
    """
    softmaxed = nn.functional.softmax(value[0], dim=1)
    samples = torch.tensor(
        np.array(
            list(
                torch.utils.data.WeightedRandomSampler(
                    softmaxed, 10000)))).float()
    if percentile == 10000:
        percentile -= 1
    return torch.sort(samples)[0][:, percentile] + samples.mean()*.001

def sampler_binned_random(value, percentile):
    """sampler_binned with random secondary scoring.

    Args:
        value: The output of the rejection model
        percentile: the (sorted) index of the sample we use
    Returns:
        The percentile largest predicted additional error, with random
        secondary scoring.
    """
    softmaxed = nn.functional.softmax(value[0], dim=1)
    samples = torch.tensor(
        np.array(
            list(
                torch.utils.data.WeightedRandomSampler(
                    softmaxed, 10000)))).float()
    if percentile == 10000:
        percentile -= 1
    to_return = torch.sort(samples)[0][:, percentile]
    return to_return + torch.rand(to_return.shape)*.1

def main(args):
    """Generate the scoring.

    Args:
        args: Command line arguments.
    Returns:
        Generates a scoring csv
    """
    loader = mturk_dataset(
        args.gt_csv, args.mturk_csv, random_other=args.random)

    model = clickhere_cnn(render4cnn(), weights_path=Paths.clickhere_weights)

    # Load the rejection model, if applicable.
    if args.method == "daer" or args.method == "bce-only" or\
    args.method == "cx-mean" or args.method == "cx-max":
        scoring_model = RejectModel(200)

    # Otherwise, set the scoring model.
    if args.method == "entropy":
        scoring_model = entropy
    elif args.method == "softmax":
        scoring_model = maxed_softmax
    elif args.method == "sampler":
        scoring_model = sampler
    elif args.method == "best-geodesic" or\
            args.method == "best-loss" or\
            args.method == "distance" or\
            args.method == "random":
        scoring_model = None
    else:
        # Load the scoring model if the scoring model needs to be loaded.
        scoring_model.load_state_dict(torch.load(args.weights))
        scoring_model.cuda().eval()
    model.cuda().eval()

    # Open the output CSV
    with open(args.outfile, "w") as outfile:
        outfile.write("image_path,kp_class,score,azim,elev,tilt,"\
                      "keypoint_dist,gt_loss,mturk_loss,loss_delta,"\
                      "gt_geodesic,mturk_geodesic,geodesic_delta\n")

    vp_loss = SoftmaxVPLoss()
    with torch.no_grad():
        for i in range(len(loader)):
            wandb.log({"score_csv_pct":float(i)/len(loader)})
            image, azim, elev, tilt, kpc_vec, obj_class, kp_map_gt,\
                    kp_map_turk, kp_dist, image_path, _, _ =\
                    loader[i]
            kpc_vec = torch.tensor(kpc_vec).detach().unsqueeze(0)
            image = image.unsqueeze(0).cuda()
            kp_map_gt = kp_map_gt.unsqueeze(0)
            kp_map_turk = kp_map_turk.unsqueeze(0)

            # If our method uses a deep rejection model, run that pass
            if args.method != "entropy" and args.method != "sampler"\
            and args.method != "distance" and args.method != "best-loss"\
            and args.method != "best-geodesic" and args.method != "random"\
            and args.method != "softmax":
                score = scoring_model(image.cuda().float(),\
                                      kp_map_turk.cuda().float(),\
                                      kpc_vec.cuda().float())

            # Perform sampling methods if applicable.
            if args.method == "daer":
                cx_softmaxed = score['output_cx'].softmax(dim=1)
                values = torch.arange(
                    cx_softmaxed[0, :].shape[0]).cuda().float()
                score = (values * cx_softmaxed).sum(dim=1) * score['output_bce']
                #score = score['output_cx'].argmax(dim=1).float() * \
                #        score['output_bce']

            if args.method == "cx-mean":
                cx_softmaxed = score['output_cx'].softmax(dim=1)
                values = torch.arange(
                    cx_softmaxed[0, :].shape[0]).cuda().float()
                score = (values * cx_softmaxed).sum(dim=1)

            if args.method == "cx-max":
                score = score['output_cx'].argmax(dim=1).float()

            if args.method == "bce-only":
                score = score['output_bce']

            if args.method == "binned_rand":
                if args.percentile > 0:
                    score = sampler_binned_random(
                        [score], args.percentile*100)
                else:
                    score = score.argmax()

            # Find errors
            image = image.cuda().repeat(2, 1, 1, 1)
            kp_maps = torch.cat((kp_map_gt, kp_map_turk), dim=0).cuda()
            kpc_vec = kpc_vec.cuda().repeat(2, 1).float()
            obj_class = torch.tensor(obj_class).cuda().repeat(2)
            output = model(image, kp_maps, kpc_vec, obj_class)

            if args.method == "entropy":
                score = scoring_model(output)[1]
            elif args.method == "sampler":
                score = scoring_model(output, args.percentile*100)[1]
            elif args.method == "softmax":
                score = scoring_model(output)

            total_vp_loss = \
                    vp_loss(
                        output[0], torch.tensor(azim).unsqueeze(0).repeat(2),\
                        do_reduce=False)+\
                    vp_loss(
                        output[1], torch.tensor(elev).unsqueeze(0).repeat(2),\
                        do_reduce=False)+\
                    vp_loss(
                        output[2], torch.tensor(tilt).unsqueeze(0).repeat(2),\
                        do_reduce=False)

            azim_guesses = output[0].argmax(dim=1).float()
            elev_guesses = output[1].argmax(dim=1).float()
            tilt_guesses = output[2].argmax(dim=1).float()
            azim_expanded = torch.tensor(azim).repeat(2).float()
            elev_expanded = torch.tensor(elev).repeat(2).float()
            tilt_expanded = torch.tensor(tilt).repeat(2).float()

            geodesic_err = compute_angle_dists(
                [azim_guesses, elev_guesses, tilt_guesses],\
                [azim_expanded, elev_expanded, tilt_expanded])*180/np.pi

            if args.method == "distance":
                score = kp_dist
            elif args.method == "best-geodesic":
                score = geodesic_err[1]-geodesic_err[0]
            elif args.method == "best-loss":
                score = total_vp_loss[1]-total_vp_loss[0]
            elif args.method == "random":
                score = torch.rand(1)

            # Write this line to a CSV
            with open(args.outfile, "a") as outfile:
                outfile.write(image_path+","+str(kpc_vec[0, :].argmax().item())\
                              +","+str(score.item())+","+str(azim)+","\
                              +str(elev)+","+str(tilt)+","+str(kp_dist)+","\
                              +str(total_vp_loss[0].item())+","+\
                              str(total_vp_loss[1].item())+","+\
                              str((total_vp_loss[1]-total_vp_loss[0]).item())\
                              +","+str(geodesic_err[0].item())+","+\
                              str(geodesic_err[1].item())+","\
                              +str((geodesic_err[1]-geodesic_err[0]).item())\
                              +"\n")

        wandb.save(args.outfile)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()

    # logging parameters
    PARSER.add_argument('--gt_csv', type=str, default=None, required=True)
    PARSER.add_argument('--mturk_csv', type=str, default=None, required=True)
    PARSER.add_argument('--outfile', type=str, default=None, required=True)
    PARSER.add_argument('--method', type=str, default=None, required=True)
    PARSER.add_argument('--weights', type=str, default=None, required=True)
    PARSER.add_argument('--random', action="store_true", default=False)

    CMD_ARGS = PARSER.parse_args()

    main(CMD_ARGS)
