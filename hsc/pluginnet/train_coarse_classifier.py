"""Train a classifier to perform the coarse classification on SUN397.

This script trains a classifier to perform coarse classification on SUN397,
which will be used as evidence for the partial evidence task.

Example:
    Similar to the instructions in the readme, train_coarse_classifier.py
    should be run from the workspace/sun397 folder. The command is:
        python ../../pluginnet/train_coarse_classifier.py output/[]
"""
import argparse
import os
import json
import time
import torch
from torchvision.models import resnet18
from pluginnet.sun397.testing import create_mode_pe as sun397_create_mode_pe
from pluginnet.sun397.testing import create_mode_base as sun397_create_mode_base
import wandb
from get_target_tensor import get_target_tensor

#pylint: disable=invalid-name, no-member

tasks = {'sun397_pe': sun397_create_mode_pe, 'sun397': sun397_create_mode_base}

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

def train_step(model, data_loader, optimizer, device="cuda"):
    """Performs a training step (epoch)

    Args:
        model: The model which performs coarse classification
        data_loader: The dataloader.
        optimizer: The optimizer
        device: run on cuda or cpu?

    Returns:
        Mean loss and accuracy.
    """
    # initialize the loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # track the running loss and accuracy
    running_loss = 0
    samples_correct = 0
    num_samples = 0

    for _, ((target, image), _) in enumerate(data_loader):
        # Move information to the device.
        image = image.to(device)
        target = target.to(device)

        # Run the image through the model.
        model_output = model(image)[:, :7]
        # Count how many were classified correctly.
        target = get_target_tensor(target, device)
        samples_correct += (model_output.argmax(dim=1) == target).float().sum()

        # Calculate the loss
        loss = loss_fn(model_output, target)

        # save information for logging
        running_loss += loss.item() * target.shape[0]
        num_samples += target.shape[0]

        # backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # return a dict that can be passed to wandb
    return {'train_loss':running_loss/num_samples,\
            'train_accuracy': samples_correct/num_samples}

def val_step(model, data_loader, device="cuda"):
    """Performs a validation step

    Args:
        model: The model which performs coarse classification
        data_loader: The dataloader.
        device: run on cuda or cpu?

    Returns:
        Mean loss and accuracy.
    """
    # initialize the loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # track the running loss for logging
    running_loss = 0
    samples_correct = 0
    num_samples = 0

    for _, ((target, image), _) in enumerate(data_loader):
        # Move information to the device
        image = image.to(device)
        target = target.to(device)

        # Perform a forward pass
        model_output = model(image)[:, :7]

        # Convert the target to an index.
        target = get_target_tensor(target, device)

        # How many guesses were correct?
        samples_correct += (model_output.argmax(dim=1) == target).float().sum()

        # Calculate the loss
        loss = loss_fn(model_output, target)

        # save information for logging
        running_loss += loss.item() * target.shape[0]
        num_samples += target.shape[0]

    # Return dict for wandb
    return {'val_loss':running_loss/num_samples,\
            'val_accuracy': samples_correct/num_samples}

def main():
    """Runs the training.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Train a rejection model.')
    parser.add_argument('model_dir')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('-e', '--num_epochs', default=20, type=int)
    parser.add_argument('-v', '--val_step', default=1, type=int)
    parser.add_argument('-s', '--save_epoch', default=5, type=int)
    parser.add_argument('--start_weights', default=None, type=str)
    parser.add_argument('--lr', default=1e-3, type=float)

    args = parser.parse_args()

    # initialize wandb
    wandb.init(project="gtd-hsc", config=args.__dict__,\
               name="coarse-"+str(args.lr)+"-"+str(time.time()))

    # create output directory for saving.
    try:
        output_dir = os.path.join(
            "output", wandb.run.project, wandb.run.id  + "-" + wandb.sweep.id)
    except AttributeError:
        output_dir = os.path.join("output", wandb.run.project, wandb.run.id)

    os.makedirs(output_dir)

    # So that wandb saves the output dir
    args.output_dir = output_dir

    # Setup
    # This is entirely from the original codebase.
    conf = load_conf(os.path.join(args.model_dir, 'conf.json'))
    conf['split'] = "val"
    conf['base_model_file'] = os.path.join(
        args.model_dir, 'model_best_state_dict.pth.tar')
    val_set, _, _, _ = tasks[conf['task']](conf)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.batch_size, pin_memory=torch.cuda.is_available(), \
        drop_last=False)

    # Maybe not the most efficient way to get the dataset, but it works.
    conf['split'] = "train"
    train_set, _, _, _ = tasks[conf['task']](conf)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, \
        num_workers=args.workers, pin_memory=torch.cuda.is_available(),\
        drop_last=False)

    # Set the device.
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # Initialize the model. We use the full pretrained resnet,
    # but only access the first 3 outputs.
    model = resnet18(pretrained=True).cuda()

    # Load start weights if you can
    if args.start_weights is not None:
        model.load_state_dict(torch.load(args.start_weights))

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    max_acc = 0 # Used for saving best weights.
    for epoch in range(args.num_epochs):

        # If we are at a save epoch, save.
        if epoch % args.save_epoch == 0 and epoch > 0:
            torch.save(model.state_dict(),\
                       os.path.join(output_dir, str(epoch)+".weights"))

        # If we are at a validation epoch, validate.
        if epoch % args.val_step == 0:
            model.eval()
            with torch.no_grad():
                val_return = val_step(
                    model, val_loader, device=device)
            val_return['epoch'] = epoch
            wandb.log(val_return)

            # Save the most accurate model.
            if val_return['val_accuracy'] > max_acc:
                max_acc = val_return['val_accuracy']
                torch.save(model.state_dict(),\
                           os.path.join(output_dir, "model_best_acc.weights"))

        # Take a train step.
        model.train()
        train_return = train_step(model, train_loader, optimizer, device=device)
        train_return['epoch'] = epoch
        print(train_return)
        wandb.log(train_return)

if __name__ == '__main__':
    main()
