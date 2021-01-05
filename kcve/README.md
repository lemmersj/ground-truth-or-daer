# Keypoint-Conditioned Viewpoint Estimation
This folder contains the code for seed rejection on the keypoint-conditioned viewpoint estimation task. The code was based on the PyTorch implementation of ClickHere CNN by Mohamed El Banani found [here](https://github.com/mbanani/pytorch-clickhere-cnn).

## Setting up the environment
Included in this folder is an exported conda environment (environment.yml). Use the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) to set up the environment. Activate it with the command:

    conda activate gtd-kcve

[Weights and Biases](https://www.wandb.com), which this code uses for experiment logging, will need to be installed via pip, as no Conda package exists:

    pip install wandb

## Generating the Synthetic Dataset
The synthetic dataset can be generated through the code provided by the original authors of ClickHere CNN [here](https://github.com/rszeto/click-here-cnn). 

When I rendered the dataset, there were some issues with reprojecting the 3D keypoints back into the image, which was likely due to using a different version of Blender, python, numpy, or some combination of those. After rendering the dataset, I recommend checking that most of the keypoints are correct using the included `check_kp_locs.py` script---this script was created quickly for debugging this issue, so you will need to change a lot of hard-coded filepaths:

    python check_kp_locs.py /path/to/dataset/root

## Training the seed rejection model
1. Download the [ClickHere CNN weights](https://drive.google.com/drive/folders/1IwUOCCwhfAG0mGOysIAL0RCvN8rsJ-Z0?usp=sharing) to the model_weights folder. CH-CNN can also be trained using the [original PyTorch code](https://github.com/mbanani/pytorch-clickhere-cnn).
2. Set paths in util/Paths.py
3. Train the model on synthetic and real data:

	    python train_reject_model.py --batch_size 512 --dataset both --eval_epoch 1 --finetune false --lr 1e-4 --mturk_csv csv_files/val_x.csv --num_bins 200 --num_epochs 201 --output_dir output patience 5 --save_epoch 5 --val_csv csv_files/veh_pascal3d_kp_valid.csv
    
4. Finetune the model on the PASCAL3D+ Dataset

    python train_reject_model.py --batch_size 512 --dataset pascalVehKP --lr 1e-4 --mturk_csv csv_files/val_x.csv --num_bins 200 --output_dir output --save_epoch 10 --patience 100 --val_csv csv_files/veh_pascal3d_kp_valid.csv --num_epochs 1000  --eval_epoch 1 --start_weights output/gtd-kcve/<run_id>/checkpoint_best_loss.pt --soft_target True

5. Run the evaluation

    python run_analysis.py --mturk_csv csv_files/test_x.csv --method [daer, distance, entropy, softmax, sampler]  --dir [unique output directory] --gt_csv csv_files/veh_pascal3d_kp_valid.csv [--percentile XX] [--weights output/gtd-kcve/<run_id>/checkpoint_best_loss.pt]



