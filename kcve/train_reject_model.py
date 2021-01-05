""" Train reject model for click-here cnn.
"""
import argparse
import os
import time
import sys
import signal
from distutils.util import strtobool
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from util import Paths, get_data_loaders, metrics
from models import clickhere_cnn, render4cnn
import wandb
from early_stopping_module import EarlyStopping
from reject_model import RejectModel
from datasets.mturk_dataset import mturk_dataset

#pylint:disable=too-many-locals, no-member, too-many-branches
#pylint:disable=consider-using-enumerate
def main(args):
    """The main function.

    Args:
        args: the command line arguments.

    Returns:
        Nothing
    """

    # If we are restarting, load the previously saved epoch.
    # Otherwise, the current (start) epoch is zero.
    start_epoch = 0
    if args.last_checkpoint is not None:
        loaded_checkpoint = torch.load(args.last_checkpoint)
        start_epoch = loaded_checkpoint['epoch']

    # Create the reject model
    rejection_model = RejectModel(args.num_bins)
    rejection_model = rejection_model.cuda()

    # Get the data loaders. We only use train.
    train_loader, _ = get_data_loaders(dataset=args.dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers)

    # Initialize the (click-here cnn) task model.
    task_model = clickhere_cnn(
        render4cnn(), weights_path=Paths.clickhere_weights)

    # Create the optimizer.
    optimizer = torch.optim.Adam(
        rejection_model.get_parameters(args.finetune), lr=args.lr)

    # If we are resuming, load the optimizer.
    if args.last_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint['optimizer'])

    # If we are doing cosine annealing (or just cosine scheduling),
    # create the scheduler.
    if args.annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.t_0, eta_min=args.min_lr, last_epoch=start_epoch-1)
    else:
        scheduler = None

    # Load the weights if start weights are given.
    # TODO: Should I add a check in case a checkpoint is given as well?
    if args.start_weights is not None:
        loaded_weights = torch.load(args.start_weights)
        rejection_model.load_state_dict(loaded_weights)

    # Load the weights if a checkpoint is given.
    # This will override the weights loaded with the "start_weights" argument.
    if args.last_checkpoint is not None:
        rejection_model.load_state_dict(loaded_checkpoint['rejection_model'])

    # train/evaluate on GPU
    task_model = task_model.cuda()

    # initialize the early stopping tracker
    early_stopping = EarlyStopping(out_dir=args.out_dir,\
                                   patience=args.patience, verbose=True)

    # Create the mturk loader dataset.
    mturk_loader = torch.utils.data.DataLoader(
        dataset=mturk_dataset(
            args.val_csv, args.mturk_csv, random_other=args.random_eval),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Loop through num_epochs epochs
    for epoch in range(start_epoch, args.num_epochs):
        # eval on first epoch and on every eval_epoch afterwards.
        if epoch % args.eval_epoch == 0:
            with torch.no_grad():
                eval_step(args=args, rejection_model=rejection_model,
                          task_model=task_model, early_stopping=early_stopping,
                          epoch=epoch, loader=mturk_loader)

        # Train.
        train_step(args=args, rejection_model=rejection_model,
                   task_model=task_model, train_loader=train_loader,
                   optimizer=optimizer, epoch=epoch, scheduler=scheduler)

        # save the state every epoch (can always resume from previous epoch)
        torch.save(
            {'rejection_model': rejection_model.state_dict(),
             'epoch': epoch,
             'optimizer': optimizer.state_dict(),
             'args': args},
            args.out_dir+'/last_step.chkpt')
        if epoch % args.save_epoch == 0 and epoch > 0:
            torch.save(
                rejection_model.state_dict(),
                args.out_dir+'/epoch-'+str(epoch)+".weights")

def sampler(value, num_samples=10000):
    """Sample a rejection model output

    Args:
        value: The output of the rejection model.
        num_samples: The number of samples to take.

    Returns:
        A set of samples for the output.
    """

    # Softmax the model output, if not already.
    # TODO: Add check to see if already softmaxed?
    softmaxed = nn.functional.softmax(value, dim=1)

    # Take weighted samples.
    samples = torch.zeros(softmaxed.shape[0], num_samples)
    for i in range(softmaxed.shape[0]):
        samples[i, :] = torch.tensor(np.array(list(
            torch.utils.data.WeightedRandomSampler(
                softmaxed[i, :], num_samples)))).float()

    # Sort the samples, then add the mean to the samples as a tiebreaker.
    sorted_samples = torch.sort(samples)[0]
    return sorted_samples + \
            samples.mean(dim=1).unsqueeze(dim=1).repeat(
                1, sorted_samples.shape[1])*.001

def eval_step(args, rejection_model, task_model, early_stopping, epoch, loader):
    """Do an evaluation step.

    Args:
        args: The command line arguments.
        rejection_model: The rejection model.
        task_model: The task model.
        early_stopping: the early stopping module.
        epoch: The current epoch
        loader: The dataloader

    Returns:
        Nothing, logged to WandB.
    """

    # Both models into eval mode.
    task_model.eval()
    rejection_model.eval()

    # The score tensor keeps track of the score at various percentiles
    # additional error is the last column
    score_tensor = torch.zeros((0, 7))
    plot_tensor = torch.zeros((0, 3))
    predictor_correct_count = 0

    loss_sum = 0
    bce_loss_sum = 0
    cx_loss_sum = 0

    #for i, (images, azim, elev, tilt, obj_cls, kp_map_gt, \
    #        kp_map_turk, kp_class, _, _, _, _, _)\
    #        in enumerate(loader):
    for _ in range(args.num_evals):
        for i, (images, azim, elev, tilt, kp_class, obj_cls, kp_map_gt, \
                kp_map_turk, _, _, _, _) in enumerate(loader):
            # Move everything to CUDA.
            images = images.cuda().float()
            kp_map_turk = kp_map_turk.cuda().float()
            kp_map_gt = kp_map_gt.cuda().float()
            kp_class = kp_class.cuda().float()

            # Get the output predictions
            rejection_model_output = rejection_model(
                images, kp_map_turk, kp_class)
            cx_softmaxed = rejection_model_output['output_cx'].softmax(dim=1)
            values = torch.arange(
                cx_softmaxed[0, :].shape[0]).unsqueeze(0).repeat(
                    cx_softmaxed.shape[0], 1).cuda().float()

            if not args.bce_only:
                score_mean = (values * cx_softmaxed).sum(dim=1)
                score_max = torch.argmax(cx_softmaxed, dim=1).float()
            else:
                score_mean = rejection_model_output['output_bce']
                score_max = rejection_model_output['output_bce']

            # Get the current scores at various percentiles
            score = sampler(rejection_model_output['output_cx'])
            score_70 = score[:, 699]
            score_80 = score[:, 799]
            score_90 = score[:, 899]
            score_100 = score[:, 999]

            # Get the outputs
            gt_chcnn_out = task_model(images, kp_map_gt, kp_class, obj_cls)
            mturk_chcnn_out = task_model(images, kp_map_turk, kp_class, obj_cls)

            # Find the errors for both the mturk and gold-standard keypoints
            candidate_guess = (mturk_chcnn_out[0].argmax(dim=1).float(),
                               mturk_chcnn_out[1].argmax(dim=1).float(),
                               mturk_chcnn_out[2].argmax(dim=1).float())

            geodesic_err_candidate = \
                    metrics.compute_angle_dists(
                        candidate_guess, (
                            azim.float(), elev.float(), tilt.float()))*180/np.pi

            true_guess = (gt_chcnn_out[0].argmax(dim=1).float(),\
                               gt_chcnn_out[1].argmax(dim=1).float(),\
                               gt_chcnn_out[2].argmax(dim=1).float())

            geodesic_err_true_output = \
                    metrics.compute_angle_dists(
                        true_guess, (
                            azim.float(), elev.float(), tilt.float()))*180/np.pi

            # Calculate the additional error.
            additional_error = nn.functional.relu(geodesic_err_candidate -\
                    geodesic_err_true_output).cpu()

            # Add the current scores to the tracking tensor
            cat_tensor = torch.cat((score_mean.unsqueeze(1).cpu(),\
                                    score_max.unsqueeze(1).cpu(),\
                                    score_70.unsqueeze(1).cpu(),\
                                    score_80.unsqueeze(1).cpu(),\
                                    score_90.unsqueeze(1).cpu(),\
                                    score_100.unsqueeze(1).cpu(),\
                                    additional_error.unsqueeze(1)), dim=1)

            if not args.cx_only or not args.bce_only:
                cat_tensor[:, :cat_tensor.shape[1]-1] = \
                        cat_tensor[:, :cat_tensor.shape[1]-1].cuda() *\
                        rejection_model_output['output_bce'].unsqueeze(
                            1).repeat(1, cat_tensor.shape[1]-1)
            score_tensor = torch.cat((score_tensor, cat_tensor), dim=0)

            # Add the bce, cx, and ae to a separate tensor for plotting
            cat_plot_tensor = torch.cat((
                rejection_model_output['output_bce'].unsqueeze(1).cpu(),
                rejection_model_output['output_cx'].argmax(dim=1)\
                .unsqueeze(1).float().cpu(),
                additional_error.unsqueeze(1)), dim=1)
            plot_tensor = torch.cat((plot_tensor, cat_plot_tensor))

            # Is the classifier correct?
            # Unfortunately doesn't compensate for bias/adjustable threshold.
            predictor_correct = ((
                rejection_model_output['output_bce'] > 0.5).float() ==\
                    additional_error.cuda()).float().sum()

            predictor_correct_count += predictor_correct.item()

            loss = rejection_model.loss(
                rejection_model_output, additional_error, args.soft_target)

            # cross entropy is conditioned on the additional error not being 0
            # unless we are doing cx only
            if not args.cx_only:
                loss['cx_loss'] = (additional_error != 0).float()*\
                        loss['cx_loss'].cpu()

            # Save the individual loss components
            bce_loss_sum += loss['bce_loss'].sum().item()
            cx_loss_sum += loss['cx_loss'].sum().item()

            if args.cx_only:
                loss_sum += loss['cx_loss'].sum().item()
            elif args.bce_only:
                loss_sum += loss['bce_loss'].sum().item()
            else:
                loss_sum += args.scale*loss['bce_loss'].sum().item() +\
                loss['cx_loss'].sum().item()

    # Plot the scatterplots (regressor) and boxplots (classifier)
    where_not_zero = torch.where(plot_tensor[:, 2] != 0)
    where_zero = torch.where(plot_tensor[:, 2] == 0)
    fig_regression = plt.figure(0)
    cur_ax = fig_regression.gca()
    cur_ax.scatter(plot_tensor[where_not_zero[0], 1],
                   plot_tensor[where_not_zero[0], 2])
    cur_ax.set_xlim(0, 200)
    cur_ax.set_ylim(0, 200)

    fig_classification = plt.figure(1)
    cur_ax = fig_classification.gca()
    cur_ax.boxplot([plot_tensor[where_zero[0], 0].numpy(),\
            plot_tensor[where_not_zero[0], 0].numpy()])
    cur_ax.set_ylim(0, 1)

    # Log the various losses and plots
    to_log = {}
    to_log['test/loss_total'] = loss_sum
    to_log['test/loss_cx'] = cx_loss_sum
    to_log['test/loss_bce'] = bce_loss_sum
    to_log['test/regression'] = wandb.Image(fig_regression)
    to_log['test/classification'] = wandb.Image(fig_classification)

    # calculate the AUAER for all percentiles, starting with the median.
    titles = ['test/mean', 'test/max', 'test/70th', 'test/80th', 'test/90th',\
              'test/100th']

    # We also track the best performer.
    min_auc = 100000

    # Loop through all the percentiles.
    for which_percentile in range(score_tensor.shape[1]-1):
        # Sort based on the scores.
        sorted_indices = torch.argsort(score_tensor[:, which_percentile])

        # Remember the last column of the score tensor is the additional error.
        sorted_additional_errors = score_tensor[sorted_indices, 6]
        sorted_list = list(sorted_additional_errors.numpy())

        current_additional_risk_loss = 0
        running_auc_loss = 0
        for i in range(len(sorted_list)):
            # Coverage increases by 1/samples for every sample.
            coverage = (1.+i)/len(sorted_list)

            current_additional_risk_loss += 1./len(
                sorted_list) * sorted_list[i]

            running_auc_loss += current_additional_risk_loss/coverage\
                    * 1./len(sorted_list)
        to_log[titles[which_percentile]] = running_auc_loss
        if running_auc_loss < min_auc:
            min_auc = running_auc_loss

    # pass the loss to the early stopping module
    early_stopping(loss_sum, rejection_model)
    if early_stopping.early_stop:
        print("Ran out of patience")
        sys.exit()

    # Save the epoch, auc, and accuracy.
    to_log['test/best_auc'] = min_auc
    to_log['test/predictor_correct'] = \
            predictor_correct_count/score_tensor.shape[0]
    to_log['epoch'] = epoch
    wandb.log(to_log)

    # Close the plots
    plt.close(fig_regression)
    plt.close(fig_classification)

def train_step(args, rejection_model, task_model, train_loader,\
               optimizer, epoch, scheduler=None):
    """Train the rejection model for an epoch.

    args:
        args: The command line arguments.
        rejection_model: The rejection model.
        task_model: The task model.
        train_loader: The train loader.
        optimizer: The optimizer.
        epoch: Current epoch.
        scheduler: the LR scheduler

    returns:
        Nothing.
    """

    # Set the models to train/eval as appropriate.
    rejection_model.train()
    task_model.eval()

    # Track the loss
    loss_sum = 0.
    bce_loss_sum = 0.
    cx_loss_sum = 0.
    item_count = 0

    # and track scores for expected additional risk.
    tensor_for_ear = torch.zeros(0, 2)
    plot_tensor = torch.zeros(0, 3)

    # How often our classifier predicts correctly.
    predictor_correct_count = 0
    # How often a classifier that always guesses the higher prior is correct.
    bias_predictor_correct_count = 0

    # There's a reason the last learning rate is a list. I don't remember it.
    # Regardless, we track the learning rate.
    last_lr = [0]

    # Loop through the dataloader.
    for i, (images, azim, elev, tilt, obj_cls, kp_map_gt, \
            kp_map_other, kp_class, _, _, _, _, _)\
            in enumerate(train_loader):

        # Move everything to CUDA.
        images = images.cuda()
        kp_map_gt = kp_map_gt.cuda()
        kp_map_other = kp_map_other.cuda()
        obj_cls = obj_cls.cuda()
        kp_class = kp_class.cuda()

        # Track number of samples in epoch.
        item_count += images.shape[0]

        # Calculate the additional error.
        with torch.no_grad():

            # Get the output of the task model.
            true_output = task_model(
                images, kp_map_gt, kp_class, obj_cls)
            candidate_output = task_model(
                images, kp_map_other, kp_class, obj_cls)

            # Calculate the errors.
            candidate_guess = (candidate_output[0].argmax(dim=1).float(),\
                               candidate_output[1].argmax(dim=1).float(),\
                               candidate_output[2].argmax(dim=1).float())

            err_candidate = \
                    metrics.compute_dist_from_eye(
                        candidate_guess, (azim, elev, tilt))

            true_guess = (true_output[0].argmax(dim=1).float(),
                          true_output[1].argmax(dim=1).float(),
                          true_output[2].argmax(dim=1).float())

            err_true_output = \
                    metrics.compute_dist_from_eye(
                        true_guess, (azim, elev, tilt))

            # dist_from_eye is 0-2sqrt(2), (but the it's divided by sqrt2 in
            # the function call. Multiplying by 50 fits in our 200 bins.
            err_candidate = err_candidate*50
            err_true_output = err_true_output*50
            additional_error = err_candidate -\
                err_true_output

        # Forward pass of the rejection model.
        rejection_model_output = rejection_model(
            images, kp_map_other, kp_class)

        # Use the mean as a score.
        cx_softmaxed = rejection_model_output['output_cx'].softmax(dim=1)
        values = torch.arange(cx_softmaxed[0, :].shape[0]).unsqueeze(0).repeat(
            cx_softmaxed.shape[0], 1).cuda().float()
        scores = (values * cx_softmaxed).sum(dim=1) *\
                rejection_model_output['output_bce']

        # Track the scores and additional errors.
        scores_ar_cat = torch.cat(
            (scores.cpu().unsqueeze(1), torch.nn.functional.relu(
                additional_error.cpu().unsqueeze(1))), dim=1).detach()
        tensor_for_ear = torch.cat((tensor_for_ear, scores_ar_cat), dim=0)

        # Track the BCE, CX, and AE for plotting.
        cat_plot_tensor = torch.cat((
            rejection_model_output['output_bce'].unsqueeze(1).detach(),
            rejection_model_output['output_cx'].argmax(dim=1)\
            .unsqueeze(1).float(),
            additional_error.unsqueeze(1)), dim=1)
        plot_tensor = torch.cat((plot_tensor, cat_plot_tensor.cpu()))

        # Track the "correct" predictor's accuracy.
        predictor_correct = ((
            rejection_model_output['output_bce'] > 0.5).float() ==\
                additional_error.cuda()).float().sum()
        predictor_correct_count += predictor_correct.item()

        # Track the bias (i.e., always zero) predictor's accuracy.
        bias_predictor_correct = (
            torch.zeros(
                rejection_model_output['output_bce'].shape).cuda().float() ==\
                additional_error.cuda()).float().sum()
        bias_predictor_correct_count += bias_predictor_correct.item()

        # Calculate the loss.
        loss = rejection_model.loss(rejection_model_output, additional_error,
                                    args.soft_target)

        # cross entropy is conditioned on the additional error not being 0
        # Unless we only use cross entropy.
        if not args.cx_only:
            loss['cx_loss'] = (additional_error != 0).float()*loss['cx_loss']

        # Save the individual loss components
        bce_loss_sum += loss['bce_loss'].sum().item()
        cx_loss_sum += loss['cx_loss'].sum().item()

        # The loss depends on how we're training...
        if args.cx_only:
            loss = loss['cx_loss'].sum()
        elif args.bce_only:
            loss = loss['bce_loss'].sum()
        else:
            loss = args.scale*loss['bce_loss'].sum() + loss['cx_loss'].sum()

        # Perform an optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the LR scheduler
        if scheduler is not None:
            scheduler.step(epoch + float(i)/len(train_loader))
            last_lr = scheduler.get_lr()

        # Delete everything.
        # I'm not sure this is necessary, but WandB sweeps tend to leave
        # things in GPU memory when killed, so I have this in here.
        del images
        del azim
        del elev
        del tilt
        del obj_cls
        del kp_map_gt
        del kp_map_other
        del kp_class

        # Save the loss.
        loss_sum += loss.item()

    where_not_zero = torch.where(plot_tensor[:, 2] != 0)
    where_zero = torch.where(plot_tensor[:, 2] == 0)
    # Now calculate the AUAER.
    numpy_for_ear = tensor_for_ear.numpy()
    numpy_for_ear = numpy_for_ear[numpy_for_ear[:, 0].argsort()]

    # Additional errror risk at every point
    cumulative_sum = np.cumsum(numpy_for_ear[:, 1])
    cumulative_sum = 1./cumulative_sum.shape[0] * cumulative_sum
    coverage = (np.arange(cumulative_sum.shape[0])+1)/cumulative_sum.shape[0]
    allpoints_arisk = cumulative_sum/coverage

    # multiply by width at every point and sum
    au_ar = (allpoints_arisk * 1/cumulative_sum.shape[0]).sum()

    # Find the best case
    numpy_for_ear_perfect = numpy_for_ear[numpy_for_ear[:, 1].argsort()]

    # Risk at every point
    cumulative_sum_perfect = np.cumsum(numpy_for_ear_perfect[:, 1])
    cumulative_sum_perfect = 1./cumulative_sum.shape[0] * cumulative_sum_perfect
    allpoints_arisk_perfect = cumulative_sum_perfect/coverage

    # multiply by width at every point and sum
    au_ar_perfect = (allpoints_arisk_perfect * 1/cumulative_sum.shape[0]).sum()

    where_not_zero = torch.where(plot_tensor[:, 2] != 0)
    where_zero = torch.where(plot_tensor[:, 2] == 0)
    fig_regression = plt.figure(0)
    cur_ax = fig_regression.gca()
    cur_ax.scatter(plot_tensor[where_not_zero[0], 1],
                   plot_tensor[where_not_zero[0], 2])
    cur_ax.set_xlim(0, 200)
    cur_ax.set_ylim(0, 200)

    fig_classification = plt.figure(1)
    cur_ax = fig_classification.gca()
    cur_ax.boxplot([plot_tensor[where_zero[0], 0].numpy(),\
            plot_tensor[where_not_zero[0], 0].numpy()])
    cur_ax.set_ylim(0, 1)

    # Save all this to WandB
    log_dict = {'train/au_ar':au_ar, 'train/perfect_au_ar':au_ar_perfect,
                'train/au_ar_deviation':au_ar-au_ar_perfect,
                'train/loss_total':loss_sum,
                'epoch':epoch, 'train/loss_bce':bce_loss_sum,
                'train/loss_cx': cx_loss_sum,
                'train/predictor_correct': predictor_correct_count/item_count,
                'train/bias_predictor_correct':\
                bias_predictor_correct_count/item_count,
                'train/last_lr': last_lr[0],
                'train/classification': wandb.Image(fig_classification),
                'train/regression': wandb.Image(fig_regression)}

    wandb.log(log_dict)
    plt.close(fig_classification)
    plt.close(fig_regression)

def exit_gracefully(sig, frame):
    # pylint: disable=unused-argument
    """Exits the program.

    Args:
        sig: ???
        frame: ???
    Returns:
        Nothing
    """
    print("Caught signal " + str(sig))
    sys.exit()

if __name__ == '__main__':

    # Initialize the signal handler.
    # For some reason interrupts tend to cause data to be left on the gpu
    # when running with a wandb sweep agent. Hopefully this works.
    # According to their support, the agent sends a sigterm, then a sigkill
    signal.signal(signal.SIGTERM, exit_gracefully)
    signal.signal(signal.SIGINT, exit_gracefully)

    # Switch the matplotlib backend for headless use
    plt.switch_backend("Agg")

    PARSER = argparse.ArgumentParser(
        description="Train a rejection model for click-here cnn")

    # logging parameters
    PARSER.add_argument('--eval_epoch', type=int, default=5,
                        help="How many epochs between evaluations.")
    PARSER.add_argument('--num_workers', type=int, default=8,
                        help="Number of dataloader workers.")
    PARSER.add_argument('--save_epoch', type=int, default=1,
                        help="How often to save rejection model.")

    # training parameters
    PARSER.add_argument('--num_epochs', type=int, default=100,
                        help="How many epochs to train for.")
    PARSER.add_argument('--batch_size', type=int, default=64,
                        help="Batch size.")
    PARSER.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate. Max LR for cosine scheduling.")
    PARSER.add_argument('--finetune', type=str, default="False",
                        help="Train only final linear layers.")
    PARSER.add_argument('--cx_only', type=str, default="False",
                        help="Train only regressor (for ablation).")
    PARSER.add_argument('--bce_only', type=str, default="False",
                        help="Train only classifier (for ablation).")

    # Cosine scheduling
    PARSER.add_argument('--annealing', type=str, default="False",
                        help="Run cosine annealing/LR schedule.")
    PARSER.add_argument('--min_lr', type=float, default=1e-6,
                        help="Minimum LR for cosine scheduling.")
    PARSER.add_argument('--t_0', type=int, default=1,
                        help="How long the cosine scheduler \
                        runs before restarting.")

    PARSER.add_argument('--start_weights', type=str, default="NONE",
                        help="Rejection model start weights.")
    PARSER.add_argument('--dataset', type=str, default=None,
                        help="Which dataset to use.",
                        choices=["pascalVehKP", "both"])
    PARSER.add_argument('--output_dir', type=str, default=None, required=True,
                        help="Directory in which to save.")
    PARSER.add_argument('--num_bins', type=int,
                        help="How many bins in the reject model.")
    PARSER.add_argument('--patience', type=int, default=10,
                        help="How many epochs to wait for new min\
                        val before exiting.")
    PARSER.add_argument('--val_csv', type=str, default=None, required=True,
                        help="The validation (not mturk) csv.")
    PARSER.add_argument('--mturk_csv', type=str, default=None, required=True,
                        help="The mturk csv.")
    PARSER.add_argument('--scale', type=float, default=1,
                        help="How much to weight the bce loss.")
    PARSER.add_argument('--soft_target', type=str, default="False",
                        help="Use soft crossentropy target for regression.")
    PARSER.add_argument('--num_evals', type=int, default=1,
                        help="How many random selections we use for\
                        evaluation.")
    PARSER.add_argument('--last_checkpoint', type=str, default=None,
                        help="The checkpoint to load.")
    PARSER.add_argument('--random_eval', type=str, default="False",
                        help="Use random keypoints or mturk keypoints.")

    CMD_ARGS = PARSER.parse_args()
    CMD_ARGS.finetune = strtobool(CMD_ARGS.finetune)
    CMD_ARGS.soft_target = strtobool(CMD_ARGS.soft_target)
    CMD_ARGS.annealing = strtobool(CMD_ARGS.annealing)
    CMD_ARGS.cx_only = strtobool(CMD_ARGS.cx_only)
    CMD_ARGS.bce_only = strtobool(CMD_ARGS.bce_only)
    CMD_ARGS.random_eval = strtobool(CMD_ARGS.random_eval)

    if CMD_ARGS.start_weights.upper() == "NONE":
        CMD_ARGS.start_weights = None

    # make sure the output directory gets written to the wandb file.
    wandb.init(project="gtd-kcve",
               config=CMD_ARGS.__dict__,\
               name=CMD_ARGS.mturk_csv+"-"+CMD_ARGS.dataset+"-"+str(
                   time.time()))

    # Make output directories based on wandb run
    try:
        OUTPUT_DIR = os.path.join(
            CMD_ARGS.output_dir, wandb.run.project,
            wandb.sweep.id, wandb.run.id)
    except AttributeError:
        OUTPUT_DIR = os.path.join(
            CMD_ARGS.output_dir, wandb.run.project, wandb.run.id)
    CMD_ARGS.out_dir = OUTPUT_DIR
    os.makedirs(OUTPUT_DIR)

    main(CMD_ARGS)
