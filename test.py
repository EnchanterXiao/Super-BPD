import argparse
import os
import torch
import torch.nn as nn
from model import VGG16
from vis_flux import vis_flux, vis_flux_v2, label2color
from datasets import FluxSegmentationDataset
from torch.autograd import Variable
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np

DATASET = 'PascalContext'
TEST_VIS_DIR = './test_pred_flux/'
SNAPSHOT_DIR = './snapshots/'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Super-BPD Network")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset for training.")
    parser.add_argument("--test-vis-dir", type=str, default=TEST_VIS_DIR,
                        help="Directory for saving vis results during testing.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()


args = get_arguments()


def main():
    if not os.path.exists(args.test_vis_dir + args.dataset):
        os.makedirs(args.test_vis_dir + args.dataset)

    model = VGG16()

    model.load_state_dict(torch.load('PascalContext_400000.pth', map_location=torch.device('cpu')))

    model.eval()
    # model.cuda()

    # dataloader = DataLoader(FluxSegmentationDataset(dataset=args.dataset, mode='test'), batch_size=1, shuffle=False, num_workers=4)

    # for i_iter, batch_data in enumerate(dataloader):
    image_dir = '..\\video frame\*'
    image_files = sorted(glob.glob(image_dir))
    IMAGE_MEAN = np.array([103.939, 116.779, 123.675], dtype=np.float32)
    for image_path in image_files:
        image_name = image_path.split('\\')[-1].split('.')[0]
        print(image_path, image_name)
        image = cv2.imread(image_path, 1)
        vis_image = image.copy()
        # print(vis_image.shape)

        image = image.astype(np.float32)
        image -= IMAGE_MEAN
        image = image.transpose(2, 0, 1)

        # Input_image, vis_image, gt_mask, gt_flux, weight_matrix, dataset_lendth, image_name = batch_data
        # print(i_iter, dataset_lendth)
        # pred_flux = model(Input_image.cuda())

        Input_image = torch.from_numpy(image).unsqueeze(0)
        with torch.no_grad() as f:
            pred_flux = model(Input_image)
        # print(pred_flux)

        vis_flux_v2(vis_image, pred_flux, image_name, args.test_vis_dir)
        # vis_flux(vis_image, pred_flux, gt_flux, gt_mask, image_name, args.test_vis_dir + args.dataset + '/')

        # pred_flux = pred_flux.data.cpu().numpy()[0, ...]
        pred_flux = pred_flux.numpy()[0, ...]
        sio.savemat(args.test_vis_dir + args.dataset + '/' + image_name + '.mat', {'flux': pred_flux})


if __name__ == '__main__':
    main()





