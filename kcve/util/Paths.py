"""Path locations."""
import os

# pylint: disable=invalid-name

pascal3d_root = '/z/dat/PASCAL3D/'
syn_dataset_root = '/z/dat/clickhere_cnn/syn_images_cropped_bkg_overlaid'

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clickhere_weights = os.path.join(root_dir,
                                 'model_weights/clickhere_weights.weights')
