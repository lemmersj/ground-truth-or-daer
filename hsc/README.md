# Hierarchical Scene Classification
The code for hierarchical scene classification is based on the Plugin-Network architecture by Koperski et al. available [here](https://github.com/tooploox/plugin-networks).

## Setting up the workspace
Included in this folder is an exported conda environment (environment.yml). Use the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) to set up the environment. Activate it with the command:

    conda activate gtd-hsc

[Weights & Biases](http://wandb.com) and torchnet will need to be installed via pip:

    pip install wandb torchnet

   
Run the shell script to download SUN397:

    ./download_environment.sh
Download the pluginnet model we used [here](https://drive.google.com/drive/folders/1IwUOCCwhfAG0mGOysIAL0RCvN8rsJ-Z0?usp=sharing) and extract it to workspace/sun397/output/pretrained_pluginnet, or train it using the instructions on the [original repository](https://github.com/tooploox/plugin-networks).

Add the path to the HSC folder to your pythonpath:

    export PYTHONPATH=/some/path/ground-truth-or-daer/hsc:$PATH

and the path to the pluginnet folder to your path:

    export PATH=/some/path/ground-truth-or-daer/hsc/pluginnet:$PATH


## Training Seeding Model
From the workspace/sun397 folder:

    python ../../pluginnet/train_coarse_classifier.py output/pretrained_pluginnet/ --batch_size 64 --lr 1e-4 --num_epochs 20 --save_epoch 5 --val_step 1 --workers 4

## Training Rejection Model
From the workspace/sun397 folder:

### Full DAER
    python ../../pluginnet/train_rejection.py --model_dir output/pretrained_pluginnet/ --lr 1e-5

### Regression Only
    python ../../pluginnet/train_rejection_no_classifier.py --model_dir output/pretrained_pluginnet/ --lr 1e-4

### Classifier only (Same as seeding model)

    ../../pluginnet/train_coarse_classifier.py output/pretrained_pluginnet --lr 1e-4

Fine entropy and coarse entropy rely on the output of the classifier, and do not have trained models specifically for rejection.

## Testing Rejection Model
From the workspace/sun397 folder.
### Full DAER, Entropy, Optimal, Softmax Response

    python ../../pluginnet/test_rejection.py --model_dir output/pretrained_pluginnet/ --guess_weights output/gtd-hsc/[run_id]/model_best_acc.weights --rejection_weights output/gtd-hsc/

### Regression Only

    python ../../pluginnet/test_rejection_no_classifier.py --model_dir output/pretrained_pluginnet/ --guess_weights output/gtd-hsc/[run_id]/model_best_acc.weights --rejection_weights output/gtd-hsc/
