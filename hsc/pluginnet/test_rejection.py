"""Test a trainedrejection model on the plugin network.

This script uses an already trained resnet-based architecture for predicting
additional error on the fine-grained classification task with partial evidence,
as discussed by Koperski et al. in "Plugin Networks for Inference under
Partial Evidence." It tests when the seed is provided by a pretrained
classifier.

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
        rejection_model_output = rejection_model(image)[:, :14]
        rejection_classifier_output = rejection_model_output[:, 7:14]
        rejection_model_output = rejection_model_output[:, :7]

        rejection_model_output = torch.sigmoid(rejection_model_output)*\
                (1-torch.softmax(rejection_classifier_output, dim=1))
        # and use only the relevant outputs.
        rejection_model_output = rejection_model_output[
            torch.arange(guess_model_output.shape[0]), guess_model_output]

        # Save information for the au-aer calc
        scores = torch.cat((scores, rejection_model_output.cpu()))
        additional_errors = torch.cat(
            (additional_errors, additional_error.cpu()))

        # Save loss information
        num_samples += additional_error.shape[0]

    auaer_dict = calc_auaer(scores, additional_errors, "model")
    auaer_dict['coarse_acc'] = coarse_correct/num_samples

    return auaer_dict

def perform_optimal(task_model, guess_model, data_loader, device="cuda"):
    """Finds the performance of an optimal rejection model.

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
    scores = torch.zeros((0))
    additional_errors = torch.zeros(0)
    num_samples = 0

    for _, (input_data, target) in enumerate(data_loader):
        # Move information to the device.
        evidence = input_data[0].to(device)
        image = input_data[1].to(device)
        target = target.to(device).argmax(dim=1)

        # Produce the guess model's output.
        guess_model_output = guess_model(image)[:, :7]
        # and place in correct form
        guess_evidence = reverse_target_tensor(
            guess_model_output.argmax(dim=1), device)

        # Find gold standard answers
        guess_correct_evidence = task_model((evidence, image)).argmax(dim=1)
        correct_evidence_correct = (
            guess_correct_evidence == target).float()

        # Find answers with guessed evidence
        guess_guess_evidence = task_model((guess_evidence, image)).argmax(dim=1)
        guess_evidence_correct = (guess_guess_evidence == target).float()

        additional_error = torch.nn.functional.relu(
            correct_evidence_correct - guess_evidence_correct)

        # Save information for the au-aer calc
        scores = torch.cat((scores, additional_error.cpu()))
        additional_errors = torch.cat(
            (additional_errors, additional_error.cpu()))

        # Save loss information
        num_samples += additional_error.shape[0]

    return calc_auaer(scores, additional_errors, "optimal")

def perform_test_guess_sr(task_model, guess_model, data_loader, device="cuda"):
    """Performs testing for the softmax response on the guess model.

    Args:
        task_model: The model which does the fine-grained classification.
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

    for _, (input_data, target) in enumerate(data_loader):
        # Move information to the device.
        evidence = input_data[0].to(device)
        image = input_data[1].to(device)
        target = target.to(device).argmax(dim=1)

        # Produce the guess model's output.
        guess_model_output = guess_model(image)[:, :7].argmax(dim=1)

        # and place in correct form
        guess_evidence = reverse_target_tensor(guess_model_output, device)
        guess_model_output = guess_model(image)[:, :7]
        guess_magnitude = torch.nn.functional.softmax(
            guess_model_output, dim=1).max(dim=1)[0]

        # Find gold standard answers
        guess_correct_evidence = task_model((evidence, image)).argmax(dim=1)
        correct_evidence_correct = (
            guess_correct_evidence == target).float()

        # Find answers with guessed evidence
        guess_guess_score = task_model((guess_evidence, image))
        guess_guess_evidence = guess_guess_score.argmax(dim=1)
        guess_evidence_correct = (guess_guess_evidence == target).float()

        additional_error = torch.nn.functional.relu(
            correct_evidence_correct - guess_evidence_correct)

        # Save information for the au-aer calc
        scores = torch.cat((scores, -guess_magnitude.cpu()))
        additional_errors = torch.cat(
            (additional_errors, additional_error.cpu()))

        # Save loss information
        num_samples += additional_error.shape[0]

    return calc_auaer(scores, additional_errors, "guess_sr")

def perform_test_coarse_entropy(
        task_model, guess_model, data_loader, device="cuda"):
    """Performs testing for softmax response on the task model.

    Args:
        task_model: The model which does the fine-grained classification.
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

    for _, (input_data, target) in enumerate(data_loader):
        # Move information to the device.
        evidence = input_data[0].to(device)
        image = input_data[1].to(device)
        target = target.to(device).argmax(dim=1)

        # Produce the guess model's output.
        guess_model_output = guess_model(image)[:, :7]
        guess_model_softmax = torch.nn.functional.softmax(
            guess_model_output, dim=1)
        coarse_ent_score = -(
            guess_model_softmax * torch.log(guess_model_softmax)).sum(dim=1)
        guess_model_output = guess_model_output.argmax(dim=1)

        # and place in correct form
        guess_evidence = reverse_target_tensor(guess_model_output, device)
        guess_model_output = guess_model(image)[:, :7]

        # Find gold standard answers
        guess_correct_evidence = task_model((evidence, image)).argmax(dim=1)
        correct_evidence_correct = (
            guess_correct_evidence == target).float()

        # Find answers with guessed evidence
        guess_guess_score = task_model((guess_evidence, image))
        guess_guess_evidence = guess_guess_score.argmax(dim=1)
        guess_evidence_correct = (guess_guess_evidence == target).float()
        guess_guess_evidence = guess_guess_score.argmax(dim=1)

        guess_evidence_correct = (guess_guess_evidence == target).float()

        additional_error = torch.nn.functional.relu(
            correct_evidence_correct - guess_evidence_correct)

        # Save information for the au-aer calc
        scores = torch.cat((scores, coarse_ent_score.cpu()))
        additional_errors = torch.cat(
            (additional_errors, additional_error.cpu()))

        # Save loss information
        num_samples += additional_error.shape[0]

    return calc_auaer(scores, additional_errors, "ent_coarse")
def perform_test_entropy(task_model, guess_model, data_loader, device="cuda"):
    """Performs testing for softmax response on the task model.

    Args:
        task_model: The model which does the fine-grained classification.
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

    for _, (input_data, target) in enumerate(data_loader):
        # Move information to the device.
        evidence = input_data[0].to(device)
        image = input_data[1].to(device)
        target = target.to(device).argmax(dim=1)

        # Produce the guess model's output.
        guess_model_output = guess_model(image)[:, :7].argmax(dim=1)

        # and place in correct form
        guess_evidence = reverse_target_tensor(guess_model_output, device)
        guess_model_output = guess_model(image)[:, :7]

        # Find gold standard answers
        guess_correct_evidence = task_model((evidence, image)).argmax(dim=1)
        correct_evidence_correct = (
            guess_correct_evidence == target).float()

        # Find answers with guessed evidence
        guess_guess_score = task_model((guess_evidence, image))
        guess_guess_evidence = guess_guess_score.argmax(dim=1)
        guess_guess_score = torch.nn.functional.softmax(
            guess_guess_score, dim=1)
        guess_guess_score = -(guess_guess_score * torch.log(
            guess_guess_score)).sum(dim=1)

        guess_evidence_correct = (guess_guess_evidence == target).float()

        additional_error = torch.nn.functional.relu(
            correct_evidence_correct - guess_evidence_correct)

        # Save information for the au-aer calc
        scores = torch.cat((scores, guess_guess_score.cpu()))
        additional_errors = torch.cat(
            (additional_errors, additional_error.cpu()))

        # Save loss information
        num_samples += additional_error.shape[0]

    return calc_auaer(scores, additional_errors, "ent")
def perform_test_sr(task_model, guess_model, data_loader, device="cuda"):
    """Performs testing for softmax response on the task model.

    Args:
        task_model: The model which does the fine-grained classification.
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

    for _, (input_data, target) in enumerate(data_loader):
        # Move information to the device.
        evidence = input_data[0].to(device)
        image = input_data[1].to(device)
        target = target.to(device).argmax(dim=1)

        # Produce the guess model's output.
        guess_model_output = guess_model(image)[:, :7].argmax(dim=1)

        # and place in correct form
        guess_evidence = reverse_target_tensor(guess_model_output, device)
        guess_model_output = guess_model(image)[:, :7]

        # Find gold standard answers
        guess_correct_evidence = task_model((evidence, image)).argmax(dim=1)
        correct_evidence_correct = (
            guess_correct_evidence == target).float()

        # Find answers with guessed evidence
        guess_guess_score = task_model((guess_evidence, image))
        guess_guess_evidence = guess_guess_score.argmax(dim=1)
        guess_guess_score = torch.nn.functional.softmax(
            guess_guess_score, dim=1).max(dim=1)[0]

        guess_evidence_correct = (guess_guess_evidence == target).float()

        additional_error = torch.nn.functional.relu(
            correct_evidence_correct - guess_evidence_correct)

        # Save information for the au-aer calc
        scores = torch.cat((scores, -guess_guess_score.cpu()))
        additional_errors = torch.cat(
            (additional_errors, additional_error.cpu()))

        # Save loss information
        num_samples += additional_error.shape[0]

    return calc_auaer(scores, additional_errors, "sr")

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
    args = parser.parse_args()

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
    wandb.init(project="gtd-hsc", config=args.__dict__,)

    with torch.no_grad():
        #coarse_entropy_return = perform_test_coarse_entropy(
        #    task_model, guess_model, test_loader, device=device)
        #print("Coarse Entropy Performance: ")
        #print(coarse_entropy_return['au_aer'])
        #print("---")
        #wandb.log({'Coarse Entropy': coarse_entropy_return['au_aer']})
        #exit()
        entropy_return = perform_test_entropy(
            task_model, guess_model, test_loader, device=device)
        print("Entropy Performance: ")
        print(entropy_return['au_aer'])
        print("---")
        wandb.log({'Entropy': entropy_return['au_aer']})

        # Optimal performance
        optimal_return = perform_optimal(
            task_model, guess_model, test_loader, device=device)
        print("Optimal Performance: ")
        print(optimal_return['au_aer'])
        print("---")

        # Trained Rejection Model
        model_return = perform_test_model(
            task_model, rejection_model, guess_model, test_loader,\
            device=device)
        print("Trained Rejection Model: ")
        print(model_return['au_aer'])
        print("---")

        # Softmax Response on guessing model.
        sr_guess_return = perform_test_guess_sr(
            task_model, guess_model, test_loader, device=device)
        print("SR on guess:")
        print(sr_guess_return['au_aer'])
        print("---")

        # Softmax Response
        sr_return = perform_test_sr(
            task_model, guess_model, test_loader, device=device)
        print("Softmax Response: ")
        print(sr_return['au_aer'])
        print("---")


        wandb.log({'Optimal': optimal_return['au_aer'],\
                   'Trained': model_return['au_aer'],\
                   'Softmax Response': sr_return['au_aer'],\
                   'SR on Guess': sr_guess_return['au_aer'],\
                   'Coarse Accuracy': model_return['coarse_acc']})
        #plt.scatter(optimal_return['coverages'],\
        #            optimal_return['additional_error_risks'],\
        #            label="optimal")
        #plt.scatter(model_return['coverages'],\
        #            model_return['additional_error_risks'],\
        #            label="learned rejection")
        #plt.scatter(sr_return['coverages'],\
        #            sr_return['additional_error_risks'],\
        #            label="softmax response")
        #plt.scatter(sr_guess_return['coverages'],\
        #            sr_guess_return['additional_error_risks'],\
        #            label="guess softmax response")
        #plt.legend()
        #plt.show()

if __name__ == '__main__':
    main()
