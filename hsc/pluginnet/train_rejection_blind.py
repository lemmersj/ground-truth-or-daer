"""Train a rejection model on the plugin network to incorrect evidence.

This script trains a resnet-based architecture for predicting additional
error and rejecting bad seeds, on the fine-grained classification task
with partial evidence, as discussed by Koperski et al. in "Plugin Networks
for Inference under Partial Evidence."

Example:
    train_rejection.py should be run from the workspace/sun397 folder.
    The command is:

        $ python ../../pluginnet/train_rejection.py output/[]

    Where output/[] represents the trained pluginnet model.
"""
import argparse
import os
import json
import time
from distutils.util import strtobool
import torch
from torchvision.models import resnet18
import wandb
from pluginnet.sun397.testing import create_mode_pe as sun397_create_mode_pe
from pluginnet.sun397.testing import create_mode_base as sun397_create_mode_base
from get_target_tensor import reverse_target_tensor

#pylint: disable=invalid-name, no-member

tasks = {'sun397_pe': sun397_create_mode_pe, 'sun397': sun397_create_mode_base}

AE_HISTOGRAM = torch.zeros(2)

def load_conf(conf_file):
    """Loads the JSON file.

    Args:
        conf_file: The path to the JSON config file.

    Returns:
        A dict containing the information from the JSON config file.
    """
    with open(conf_file, 'r') as fd:
        conf = json.load(fd)
        return conf

def train_step(task_model, rejection_model, data_loader, optimizer,\
               args, device="cuda"):
    """Performs a training step (epoch)

    Args:
        task_model: The model which does the fine-grained classification.
        rejection_model: The model which performs the rejection.
        data_loader: The dataloader.
        args: Command line arguments.
        device: run on cuda or cpu?

    Returns:
        Mean loss. Classifier accuracy.
    """
    # values tracked per-epoch
    running_loss = 0 # loss
    num_samples = 0 # number of samples (for averaging)

    # Losses for classifier and additional error, respectively.
    bce_loss = torch.nn.BCELoss(reduction="none")

    # Move through an epoch
    for _, (input_data, target) in enumerate(data_loader):
        # Move information to the device.
        evidence = input_data[0].to(device)
        image = input_data[1].to(device)
        target = target.to(device).argmax(dim=1)

        # We initialize the loss here in case the train_all flag is set.
        loss = torch.zeros(1).to(device)

        # Set number of loops based on train_all
        loop_length = 1
        if args.train_all:
            loop_length = 7

        # Loop through as many times as necessary.
        for index in range(loop_length):
            with torch.no_grad():
                # Create the fake evidence
                # If train-all, we use the index, otherwise select randomly.
                if args.train_all:
                    fake_indices = index * torch.ones((
                        target.shape[0],)).to(device).long()
                else:
                    fake_indices = torch.randint(
                        7, (target.shape[0],)).to(device)

                fake_evidence = reverse_target_tensor(fake_indices, device)

                # Find gold standard answers
                guess_correct_evidence = task_model(
                    (evidence, image)).argmax(dim=1)
                correct_evidence_correct = (
                    guess_correct_evidence == target).float()

                # Find answers with fake evidence
                guess_fake_evidence = task_model(
                    (fake_evidence, image)).argmax(dim=1)
                fake_evidence_correct = (guess_fake_evidence == target).float()

                # Calculate Additional Error
                additional_error = torch.nn.functional.relu(
                    correct_evidence_correct - fake_evidence_correct)

            # Perform a forward pass of the rejection model.
            rejection_model_output = torch.sigmoid(rejection_model(image)[:, 0])

            # Calculate the loss function
            # Sigmoid is needed to make >= 0
            loss_bce = bce_loss(rejection_model_output, additional_error)
            loss += loss_bce.sum()

        # save loss information
        running_loss += loss.item() * additional_error.shape[0]
        num_samples += additional_error.shape[0]

        # backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return {'train_loss':running_loss/num_samples}

def calc_auaer(scores, additional_errors):
    """Calculates the area under the additional error risk curve.

    Args:
        scores: the scores on which to sort
        additional errors: the corresponding additional errors.

    Returns:
        The area under the additional error risk curve.
    """

    # Sort the additional error
    sorted_ae = additional_errors[torch.argsort(scores)]

    # Run the calculation.
    au_aer = 0
    numerator = 0
    num_elements = scores.shape[0]
    for i in range(num_elements):
        coverage = (i+1) / num_elements
        numerator += sorted_ae[i]

        this_additional_risk = (numerator / num_elements)/coverage

        au_aer += 1/num_elements * this_additional_risk

    return au_aer.item()

def val_step(task_model, rejection_model, data_loader, args, device="cuda"):
    """Performs a validation step

    Args:
        task_model: The model which does the fine-grained classification.
        rejection_model: The model which performs the rejection.
        data_loader: The dataloader.
        device: run on cuda or cpu?

    Returns:
        Mean loss, acc, and AU-AER.
    """
    del args
    bce_loss = torch.nn.BCELoss(reduction="none")

    # track the running loss for logging
    running_loss = 0
    num_samples = 0

    # save scores and additional error
    scores = torch.zeros(0)
    additional_errors = torch.zeros(0)

    for _, (input_data, target) in enumerate(data_loader):
        # Move information to the device.
        evidence = input_data[0].to(device)
        image = input_data[1].to(device)
        target = target.to(device).argmax(dim=1)

        # Find gold standard answers
        guess_correct_evidence = task_model((evidence, image)).argmax(dim=1)
        correct_evidence_correct = (
            guess_correct_evidence == target).float()
        for i in range(7):
            # Create the fake evidence
            fake_indices = i*torch.ones(target.shape[0],).to(device).long()
            fake_evidence = reverse_target_tensor(fake_indices, device)

            # Find answers with fake evidence
            guess_fake_evidence = task_model(
                (fake_evidence, image)).argmax(dim=1)
            fake_evidence_correct = (guess_fake_evidence == target).float()

            additional_error = torch.nn.functional.relu(
                correct_evidence_correct - fake_evidence_correct)

            # Perform a forward pass of the rejection model.
            rejection_model_output = rejection_model(image)[:, 0]

            # Separate the sigmoided (for loss) and weighted (for AUAER)
            rejection_model_output_only = torch.sigmoid(rejection_model_output)

            # Save information for the amae calc
            scores = torch.cat((scores, rejection_model_output.cpu()))
            additional_errors = torch.cat(
                (additional_errors, additional_error.cpu()))

            loss_bce = bce_loss(rejection_model_output_only, additional_error)
            running_loss += loss_bce.sum()
            num_samples += additional_error.shape[0]

    return {'val loss': running_loss/num_samples,
            'val auaer': calc_auaer(scores, additional_errors)}

def main():
    """Runs the training.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Train a rejection model.')
    parser.add_argument('--model_dir')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('-e', '--num_epochs', default=50, type=int)
    parser.add_argument('-v', '--val_step', default=1, type=int)
    parser.add_argument('-s', '--save_epoch', default=10, type=int)
    parser.add_argument('--start_weights', default=None, type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--important_only', type=str, default="False",\
                       help="Only train on examples where there is"\
                        "additional error.")
    parser.add_argument('--train_all', type=str, default="False")
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--run_num', required=False, type=int, default=0)
    args = parser.parse_args()

    # to use wandb sweeps, we can use "store-true" flags, so we need to
    # convert strings into the corresponding bools.
    args.important_only = strtobool(args.important_only)
    args.train_all = strtobool(args.train_all)

    # initialize wandb
    wandb.init(project="gtd-hsc", config=args.__dict__,\
               name="reject-noseed-"+str(args.lr)+"-"+str(time.time()))

    # initialize output directory for saving.
    try:
        output_dir = os.path.join(
            "output", wandb.run.project, wandb.run.id  + "-" + wandb.sweep.id)
    except AttributeError:
        output_dir = os.path.join("output", wandb.run.project, wandb.run.id)

    # So that wandb saves it
    args.output_dir = output_dir

    os.makedirs(output_dir)
    # Set the device.
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # Setup
    # This is entirely from the original codebase.
    conf = load_conf(os.path.join(args.model_dir, 'conf.json'))
    conf['split'] = "val"
    conf['base_model_file'] = os.path.join(
        args.model_dir, 'model_best_state_dict.pth.tar')
    val_set, task_model, _, _ = tasks[conf['task']](conf)

    # Move the network to the device.
    task_model = task_model.eval().to(device)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.workers, pin_memory=torch.cuda.is_available(),\
        drop_last=False)

    # Maybe not the most efficient way to get the dataset, but it works.
    conf['split'] = "train"
    train_set, _, _, _ = tasks[conf['task']](conf)

    # If we want to only work on tasks where the seed makes a difference, use
    # a different dataloader.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, \
        num_workers=args.workers, pin_memory=torch.cuda.is_available(),\
        drop_last=False)


    # Initialize the rejection model. We use the full pretrained resnet,
    # but only access the first 14 outputs.
    rejection_model = resnet18(pretrained=True).cuda()

    # Load start weights if you can
    if args.start_weights is not None:
        rejection_model.load_state_dict(torch.load(args.start_weights))

    # Initialize the optimizer
    optimizer = torch.optim.Adam(rejection_model.parameters(), lr=args.lr)


    min_auaer = 100000 # Used to save best performer.
    # Perform the training
    for epoch in range(args.num_epochs + 1):
        # If it's a designated save epoch, save.
        if epoch % args.save_epoch == 0:
            torch.save(rejection_model.state_dict(),\
                       os.path.join(output_dir, str(epoch)+".weights"))

        # If it's a designated val step, run validation.
        if epoch % args.val_step == 0:
            rejection_model.eval()
            with torch.no_grad():
                val_return = val_step(
                    task_model, rejection_model, val_loader, args,\
                    device=device)
            val_return['epoch'] = epoch
            print(val_return)
            wandb.log(val_return)
            # If the AUAER is lowest, save it.
            if val_return['val auaer'] < min_auaer:
                min_auaer = val_return['val auaer']
                torch.save(rejection_model.state_dict(),\
                           os.path.join(output_dir, "model_best_auaer.weights"))

        # Perform a train step.
        rejection_model.train()
        to_log = train_step(
            task_model, rejection_model, train_loader, optimizer,
            args, device=device)
        to_log['epoch'] = epoch
        print(to_log)
        wandb.log(to_log)

if __name__ == '__main__':
    main()
