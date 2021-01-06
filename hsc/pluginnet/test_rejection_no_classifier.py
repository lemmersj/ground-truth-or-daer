"""Test a trained rejection model on the plugin network.

This script uses an already trained resnet-based architecture for predicting
additional error on the fine-grained classification task with partial evidence,
as discussed by Koperski et al. in "Plugin Networks for Inference under
Partial Evidence." It tests when the seed is provided by a pretrained
classifier. The rejection model does not have the classifier component.

Example:
    Similar to the instructions in the readme, test_rejection.py should be run
    from the workspace/sun397 folder. The command is:
        python ../../pluginnet/test_rejection.py output/[] --guess_weights []
        --rejection_weights []
"""
import argparse
import os
import json
import torch
from torchvision.models import resnet18
import wandb
from pluginnet.sun397.testing import create_mode_pe as sun397_create_mode_pe
from pluginnet.sun397.testing import create_mode_base as sun397_create_mode_base
from get_target_tensor import reverse_target_tensor

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

def calc_auaer(scores, additional_errors, label):
    """Calculates the area under the additional error risk curve.

    Args:
        scores: the scores on which to sort
        additional errors: the corresponding additional errors.

    Returns:
        The area under the additional error risk curve, as well as AER and
        coverage.
    """

    # Sort the additional error
    sorted_ae = additional_errors[torch.argsort(scores)]

    # Run the calculation.
    au_aer = 0
    numerator = 0
    num_elements = scores.shape[0]

    coverages = []
    additional_error_risks = []
    for i in range(num_elements):
        coverage = (i+1) / num_elements
        coverages.append(coverage)

        numerator += sorted_ae[i]

        this_additional_error_risk = (numerator / num_elements)/coverage
        additional_error_risks.append(this_additional_error_risk.item())

        wandb.log(
            {"coverage":coverage, label+"_aer":this_additional_error_risk})

        au_aer += 1/num_elements * this_additional_error_risk

    return {'au_aer':au_aer, 'coverages':coverages,\
            'additional_error_risks':additional_error_risks}

def perform_test_model(task_model, rejection_model, guess_model,\
             data_loader, device="cuda"):
    """Performs testing for the rejection model.

    Args:
        task_model: The model which does the fine-grained classification.
        rejection_model: The model which performs the rejection.
        guess_model: the model which produces the partial evidence.
        data_loader: The dataloader.
        device: run on cuda or cpu?

    Returns:
        The dict from calc_auaer, including auaer, and AER-coverage curve.
    """
    # save scores and additional error
    scores = torch.zeros(0)
    additional_errors = torch.zeros(0)
    num_samples = 0
    coarse_correct = 0

    for _, (input_data, target) in enumerate(data_loader):
        # Move information to the device.
        evidence = input_data[0].to(device)
        image = input_data[1].to(device)
        target = target.to(device).argmax(dim=1)

        # Produce the guess model's output.
        guess_model_output = guess_model(image)[:, :7].argmax(dim=1)

        # and place in correct form
        guess_evidence = reverse_target_tensor(guess_model_output, device)

        # Track the accuracy of the coarse classifier
        coarse_correct += (guess_evidence == evidence).all(dim=1).float().sum()

        # Find gold standard answers
        guess_correct_evidence = task_model((evidence, image)).argmax(dim=1)
        correct_evidence_correct = (
            guess_correct_evidence == target).float()

        # Find answers with guessed evidence
        guess_guess_evidence = task_model((guess_evidence, image)).argmax(dim=1)
        guess_evidence_correct = (guess_guess_evidence == target).float()

        additional_error = torch.nn.functional.relu(
            correct_evidence_correct - guess_evidence_correct)

        # Perform a forward pass of the rejection model.
        rejection_model_output = rejection_model(image)[:, :7]

        # and use only the relevant outputs.
        rejection_model_output = rejection_model_output[
            torch.arange(guess_model_output.shape[0]), guess_model_output]

        # Save information for the au-aer calc
        scores = torch.cat((scores, 1-rejection_model_output.cpu()))
        additional_errors = torch.cat(
            (additional_errors, additional_error.cpu()))

        # Save loss information
        num_samples += additional_error.shape[0]

    auaer_dict = calc_auaer(scores, additional_errors, "model")
    auaer_dict['coarse_acc'] = coarse_correct/num_samples

    return auaer_dict

def main():
    """Runs the training.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Test a rejection model.')
    parser.add_argument('--model_dir')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--guess_weights', required=True, type=str)
    parser.add_argument('--rejection_weights', required=True, type=str)
    parser.add_argument('--coarse_idx', required=False, type=int, default=-1)
    parser.add_argument('--important_only', type=str, default="False",\
                       help="Only train on examples where there is"\
                        "additional error.")
    args = parser.parse_args()

    # Unfortunately, have to do this for the gridsearch.
    args.important_only = (
        args.important_only[0] == "T" or args.important_only[0] == "t")
    # Setup
    # Set the device.
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # This is mostly from the original codebase.
    conf = load_conf(os.path.join(args.model_dir, 'conf.json'))
    conf['split'] = "test"
    conf['base_model_file'] = os.path.join(
        args.model_dir, 'model_best_state_dict.pth.tar')
    test_set, task_model, _, _ = tasks[conf['task']](conf)
    # Move the network to the device.
    task_model = task_model.eval().to(device)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.workers, pin_memory=torch.cuda.is_available(),\
        drop_last=False)

    # Initialize the rejection model.
    rejection_model = resnet18(pretrained=False).cuda()
    rejection_model.load_state_dict(torch.load(args.rejection_weights))
    rejection_model.eval().to(device)

    # Initialize the guess model.
    guess_model = resnet18(pretrained=False).cuda()
    guess_model.load_state_dict(torch.load(args.guess_weights))
    guess_model.eval().to(device)

    # initialize wandb
    wandb.init(project="pluginnet-train-5-20-20", config=args.__dict__,)

    with torch.no_grad():
        # Trained Rejection Model
        model_return = perform_test_model(
            task_model, rejection_model, guess_model, test_loader,\
            device=device)
        print("Trained Rejection Model: ")
        print(model_return['au_aer'])
        print("---")

        wandb.log({'Trained': model_return['au_aer']})

if __name__ == '__main__':
    main()
