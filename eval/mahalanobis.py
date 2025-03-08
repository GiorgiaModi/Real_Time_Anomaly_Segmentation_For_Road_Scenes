# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import random
from PIL import Image, ImageOps
import numpy as np
import os.path as osp
from argparse import ArgumentParser
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from transform import Relabel, ToLabel, Colorize
from dataset1 import cityscapes
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ERFNet = importlib.import_module('train.erfnet').ERFNet
ERFNetTransform = importlib.import_module('train.augmentations').ERFNetTransform

# general reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 19

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/content/Validation_Dataset/RoadObsticle21/images/*.webp",
        help="A single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="/content/AnomalySegmentation") 
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--model', default="erfnet") 
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--mean', default = '') #/save/mean_cityscapes_erfnet.npy
    args = parser.parse_args()

    #modelpath = args.loadDir +"/" +args.model + ".py"
    weightspath = args.loadDir + args.loadWeights
    mean_is_computed = len(args.mean) > 0
    mean_path = args.loadDir + args.mean

    print ("Loading model: " + args.model)
    print ("Loading weights: " + weightspath)
    
    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    if mean_is_computed:
        pre_computed_mean = np.load(mean_path)
        pre_computed_mean = torch.from_numpy(pre_computed_mean).cuda()
        print(f"pre_computed_mean {pre_computed_mean.shape}")
    
    # augmentations to be applied during training
    co_transform = ERFNetTransform(False, augment=False, height=512)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=1, shuffle=True)
    
    model = ERFNet(NUM_CLASSES+1)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        print(state_dict.keys())
        print(own_state.keys())
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    

    # Covariance matrix
    cov_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.float32, device='cuda')
    num_images = 0  

    sum_per_class = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.float32, device="cpu" if args.cpu else 'cuda')
    pixel_count_per_class = torch.zeros(NUM_CLASSES, dtype=torch.int32, device="cpu" if args.cpu else 'cuda')

    for images, labels in tqdm(loader):
        if not args.cpu:
            images = images.cuda()
            #labels = labels.cuda()
           
        output = None
        with torch.no_grad():
            if args.model == "bisenet":
                result = model(images)[0].squeeze(0)
            else:
                result = model(images).squeeze(0)
            # remove last channel
            result = result[:NUM_CLASSES]
        
        # If mean is not computed, accumulate sum and count per class
        if not mean_is_computed:
            # Accumulate the sum of the output for each class
            for c in range(NUM_CLASSES):
                # Create a mask for the pixels corresponding to class `c`
                mask = (labels == c).squeeze()
                
                # Accumulate the sum of the output for class `c`
                sum_per_class[c] += torch.sum(result[:, mask], dim=1)
                
                # Accumulate the count of pixels for class `c`
                pixel_count_per_class[c] += torch.sum(mask)

        else:
            for c in range(NUM_CLASSES):
                # Create a mask for the pixels corresponding to class `c`

                mask = (labels == c).squeeze()
                # Center the output relative to the precomputed mean

                centered = result[:, mask] - pre_computed_mean[c].unsqueeze(1)
                print("centered", centered)
                cov_matrix += centered @ centered.T
            
        num_images += 1

    # After processing all images, calculate the mean per class
    if not mean_is_computed:
        for c in range(NUM_CLASSES):
            if pixel_count_per_class[c] > 0:
                sum_per_class[c] /= pixel_count_per_class[c]
        
        print(f"Mean per class: {sum_per_class.shape}")
        np.save(f"{args.loadDir}/save/mean_cityscapes_{args.model}.npy", sum_per_class.data.cpu().numpy())
        print(f"Mean output saved as '{args.loadDir}/save/mean_cityscapes_{args.model}.npy'")
    else: 
        print("cov_matrix", cov_matrix)
        cov_matrix /= (num_images*512*1024)# Normalize by the number of pixels
        print(f"Covariance matrix: {cov_matrix.shape}")
        print("cov_matrix", cov_matrix)
        np.save(f"{args.loadDir}/save/cov_cityscapes_{args.model}.npy", cov_matrix.data.cpu().numpy())
        print(f"Covariance matrice saved as '{args.loadDir}/save/cov_matrix_{args.model}.npy'")

if __name__ == '__main__':
    main()