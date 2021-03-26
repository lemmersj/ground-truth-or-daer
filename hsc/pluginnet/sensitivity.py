"""Calculate the sensitivity of the plugin network to incorrect evidence.

This script exhaustively searches all potential evidence from the SUN397
fine-grained classification task with partial evidence, as discussed by
Koperski et al. in "Plugin Networks for Inference under Partial Evidence."

Example:
    Similar to the instructions in the readme, sensitivity.py should be run
    from the workspace/sun397 folder. The command is::

        $ python ../../pluginnet/sensitivity.py output/[] test
"""
import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pluginnet.sun397.testing import create_mode_pe as sun397_create_mode_pe
from pluginnet.sun397.testing import create_mode_base as sun397_create_mode_base
import csv
import torchvision

tasks = {'sun397_pe': sun397_create_mode_pe, 'sun397': sun397_create_mode_base}

def reverse_normalize(tensor):
    out_tensor = torch.tensor(tensor)
    out_tensor[:, 0, :, :] = out_tensor[:, 0, :, :] * 0.229 
    out_tensor[:, 1, :, :] = out_tensor[:, 1, :, :] * 0.224 
    out_tensor[:, 2, :, :] = out_tensor[:, 2, :, :] * 0.225
    out_tensor[:, 0, :, :] = out_tensor[:, 0, :, :] + 0.485 
    out_tensor[:, 1, :, :] = out_tensor[:, 1, :, :] + 0.456 
    out_tensor[:, 2, :, :] = out_tensor[:, 2, :, :] + 0.406 

    return out_tensor

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

def main():
    """Runs the sensitivity analysis.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Find the sensitivity of our model to incorrect evidence.')
    parser.add_argument('model_dir')
    parser.add_argument('split', choices=['train', 'val', 'test'])
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size')

    args = parser.parse_args()

    # Setup
    # This is entirely from the original codebase.
    conf = load_conf(os.path.join(args.model_dir, 'conf.json'))
    conf['split'] = args.split
    conf['base_model_file'] = os.path.join(
        args.model_dir, 'model_best_state_dict.pth.tar')
    test_set, net, _, _ = tasks[conf['task']](conf)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=args.workers, \
        pin_memory=torch.cuda.is_available(), drop_last=False)

    # Set the device.
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # Move the network to the device.
    net = net.eval().to(device)

    # Put all the potential pieces of evidence in one tensor,
    # for one pass through the network.
    all_evidence = torch.zeros((0, 3))
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                if i == 0 and j == 0 and k == 0:
                    continue
                this_row = torch.tensor(np.array([[i, j, k]])).float()
                all_evidence = torch.cat((all_evidence, this_row), dim=0)

    all_evidence = all_evidence.to(device)

    # How many pieces of incorrect evidence cause additional error.
    causes_additional_error = 0
    # How many pieces of incorrect evidence there are.
    sample_count = 0
    # How many images respond to zero through 7 pieces of evidence.
    ae_histogram = torch.zeros(7)

    # Same as above, but only for examples where the classification is correct.
    correct_samples = 0 # Not quite the same as above I guess...
    causes_additional_error_and_correct = 0
    ae_correct_histogram = torch.zeros(7)

    write_list = []
    # Loop through all the test samples, no gradients.

    to_pil = torchvision.transforms.ToPILImage()

    with torch.no_grad():
        for dataset_index, (input_data, target) in enumerate(test_loader):

            #image_to_save = to_pil(reverse_normalize(input_data[1]).squeeze())
            #image_to_save.save("out_images/"+str(dataset_index)+".jpg")
            # Move information to the device.
            evidence = input_data[0].to(device)
            image = input_data[1].to(device)
            target = target.to(device)

            write_dict = {'index': dataset_index}
            write_dict['evidence'] = str(evidence.cpu().int().numpy())

            # Find the prediction for every piece of evidence.
            guesses = net((all_evidence, image.repeat(7, 1, 1, 1))).argmax(
                dim=1)

            # What is the correct answer?
            correct_answer = target.argmax()
            write_dict['correct_answer'] = correct_answer.item()
            for guess_idx in range(guesses.shape[0]):
                write_dict[str(all_evidence[guess_idx].int().cpu().numpy())] =\
                        guesses[guess_idx].item()

            # And do the guesses match the correct answer?
            all_correct = (correct_answer.repeat(7) == guesses).float()

            solve_as_loaded = net((evidence, image)).argmax(dim=1)

            # Is the real (i.e., with the correct evidence) answer correct?
            correct_evidence_correct = (
                solve_as_loaded == correct_answer).float()

            # Calculate additional error for all examples.
            # If the correct guess is incorrect, there will never be
            # additional error.
            # Also, the correct answer will never contribute.
            additional_error = torch.nn.functional.relu(
                correct_evidence_correct-all_correct)

            # Count the pieces of incorrect evidence that cause
            # additional error.
            causes_additional_error += additional_error.sum().item()
            # Remember how many pieces of evidence in this image cause
            # additional error.
            ae_histogram[additional_error.sum().int().item()] += 1
            # And track the total number of incorrect evidences.
            sample_count += 7

            # Keep track specifically of instances where
            # the correct answer is correct.
            # The variables are mostly the same as above.
            if correct_evidence_correct:
                causes_additional_error_and_correct += \
                        additional_error.sum().item()
                ae_correct_histogram[additional_error.sum().int().item()] += 1
                correct_samples += 1

            write_list.append(write_dict)

    # save specific output
    with open("evidence_results.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = write_list[0].keys())
        writer.writeheader()
        for row in write_list:
            writer.writerow(row)
    # Print the results.
    print("There are " + str(sample_count) + " incorrect pieces of evidence.")
    print("Of these, " + str(causes_additional_error) + \
          " cause additional error")
    print("The histogram is: " + str(ae_histogram))
    print("There are a total of " + str(correct_samples) + \
          " correctly classified samples.")
    print("Of which " + str(causes_additional_error_and_correct) + \
          " incorrect pieces of evidence cause additional error.")
    print("The histogram is: " + str(ae_correct_histogram))

    from IPython import embed
    embed()
    # and create plots
    plt.bar(np.arange(7), ae_histogram/ae_histogram.sum()*100)
    plt.xlabel("# labels which cause additional error")
    plt.ylabel("% of Images")
    plt.savefig("error-dist-all-"+str(args.split)+".eps", bbox_inches="tight")

    plt.cla()
    plt.bar(np.arange(7), ae_correct_histogram/ae_correct_histogram.sum()*100)
    plt.xlabel("# labels which cause additional error")
    plt.ylabel("% of Images")
    plt.savefig("error-dist-correctonly-"+str(args.split)+".eps", bbox_inches="tight")

if __name__ == '__main__':
    main()
