# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics
from visualization import save_colored_score_image, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

from icecream import ic
from temperature_scaling import ModelWithTemperature
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import sys
import importlib
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ERFNet = importlib.import_module('train.erfnet').ERFNet
ENet = importlib.import_module('train.enet').ENet
BiSeNetV1 = importlib.import_module('train.bisenet').BiSeNetV1


seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def mahalanobis_distance_score(output, means, cov_inv):
    """
    Compute the Mahalanobis distance-based confidence score for a test sample.
    
    f_x: test sample feature vector (numpy array of shape (20, 512, 1024))
    means: class means (numpy array of shape (20,20))
    covariance_inv: inverse of covariance matrix (numpy array of shape (20, 20))
    
    Returns the confidence score (512x1024)
    """
    
    num_features, height, width = output.shape

    # Transpose and reshape the output to handle pixels in batch
    output_flat = output.view(num_features, -1).T  # (height * width, num_features)

    # Expand means for batch computation
    means_expanded = means.unsqueeze(1)  # (num_classes, 1, num_features)
    output_expanded = output_flat.unsqueeze(0)  # (1, height * width, num_features)

    # Compute the difference (broadcasting)
    centered = output_expanded - means_expanded  # (num_classes, height * width, num_features)

    # Compute the Mahalanobis score
    scores = -torch.einsum('npi,ij,npj->np', centered, cov_inv, centered)  # (num_classes, height * width)

    # Find the maximum score for each pixel
    M_scores_flat = torch.abs(scores.max(dim=0).values)  #TODO  (height * width) era senza abs

    # Reshape to (height, width)
    M_scores = M_scores_flat.view(height, width)

    return M_scores


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/content/Validation_Dataset/RoadObsticle21/images/*.webp",
        help="A single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="/content/AnomalySegmentation/trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--model', default="erfnet") #can be erfnet, erfnet_isomaxplus, enet, bisenet
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method',default="MSP") #can be MSP or MaxLogit or MaxEntropy or Mahalanobis
    parser.add_argument('--void', action='store_true')
    parser.add_argument('--temperature', default=0) # add the path of the model absolute path
    parser.add_argument('--save-colored', type=str, default=None, help='Path where to save the image as colored score. Empty: do not save')
    parser.add_argument('--save-plots-dir', type=str, default=None, help='Directory where to save ROC and PR curves. Empty: do not save')

    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    method = args.method
    input_transform = Compose([Resize((512, 1024), Image.BILINEAR), ToTensor()])
    target_transform = Compose([Resize((512, 1024), Image.NEAREST)])

    modelpath = args.loadDir +"/" +args.model + ".py"
    weightspath = args.loadDir + args.loadWeights

    temperature = float(args.temperature)

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)
    if args.model == "erfnet":
        model = ERFNet(NUM_CLASSES).to(device)
    elif args.model == "erfnet_isomaxplus":
        model = ERFNet(NUM_CLASSES, use_isomaxplus=True).to(device)
    elif args.model =="enet":
        model = ENet(NUM_CLASSES).to(device)
    elif args.model == "bisenet":
        model = BiSeNetV1(NUM_CLASSES).to(device)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        print(state_dict.keys())
        print(own_state.keys())
        # Check if the model is 'erfnet_isomaxplus'and load the state dict for IsoMaxPlusLossFirstPart
        if args.model == "erfnet_isomaxplus" and 'loss_first_part_state_dict' in state_dict:
            # Get the state dict for IsoMaxPlusLossFirstPart
            loss_first_part_state_dict = state_dict['loss_first_part_state_dict']
            # Load the state dict for IsoMaxPlusLossFirstPart
            if hasattr(model.module.decoder, 'loss_first_part'):
                model.module.decoder.loss_first_part.load_state_dict(loss_first_part_state_dict)
            else:
                raise ValueError("IsoMaxPlusLossFirstPart not found in the model")

        if 'state_dict' in state_dict:
            load_dict = state_dict['state_dict'] 
        else:
            load_dict = state_dict

        for name, param in load_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model
    model = load_my_state_dict(model, torch.load(weightspath, map_location=device))

    if(temperature != 0 ):
        model = ModelWithTemperature(model, temperature = temperature)

    print ("Model and weights LOADED successfully")
    model.eval()
    
    if method == "Mahalanobis":
        # Load mean and covariance matrices from "save" folder
        means = np.load("/content/AnomalySegmentation/save/mean_cityscapes_erfnet.npy")
        cov = np.load("/content/AnomalySegmentation/save/cov_cityscapes_erfnet.npy")
        cov_inv = np.linalg.inv(cov)
        
        # Convert to PyTorch tensors
        means = torch.from_numpy(means).to(device)
        cov_inv = torch.from_numpy(cov_inv).to(device)
    
    for path in tqdm(glob.glob(os.path.expanduser(str(args.input)))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            if args.model == "bisenet":
                result = model(images)[0].squeeze(0)
            else:
                result = model(images).squeeze(0)

        if args.void:
            anomaly_result = F.softmax(result, dim=0)[-1]
            anomaly_result = anomaly_result.data.cpu().numpy()
        else:
            result = result[:-1] # remove the last channel (void)
            
            if(method == "MaxLogit"):
                anomaly_result = -torch.max(result, dim=0)[0]
                anomaly_result = anomaly_result.data.cpu().numpy()
            elif(method == "MaxEntropy"):
                probs = F.softmax(result, dim=0)
                entropy = torch.div(torch.sum(-probs * torch.log(probs), dim=0), torch.log(torch.tensor(probs.shape[0])))
                anomaly_result = entropy.data.cpu().numpy().astype("float32")
            elif (method == "MSP"):
                anomaly_result = 1.0 - torch.max(F.softmax(result, dim=0), dim=0)[0]
                anomaly_result = anomaly_result.data.cpu().numpy()
            elif(method == "Mahalanobis"):
                anomaly_result = mahalanobis_distance_score(result, means, cov_inv)
                anomaly_result = anomaly_result.data.cpu().numpy()
                print(f"Mahalanobis score: {anomaly_result}")
            else:
                raise ValueError("Invalid method")
             

        # Load ground truth mask
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)
        print(f"Loaded out-of-distribution ground-truths: {ood_gts}") #debug
        
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts) #ho verificato ci sono veramente dei 2 all'interno dell'immagine
        if "LostAndFound" in pathGT: # LostAndFound qui non entra
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT: #  qui non entra
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask, images
        torch.cuda.empty_cache()

    #file.write( "\n")

    # Calculate metrics: AUPRC and FPR@TPR95
    ood_gts = np.array(ood_gts_list) 
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1) #array con True dove ood_gts == 1 resto false
    ind_mask = (ood_gts == 0) #array con True dove ood_gts == 0 resto false

    ood_out = anomaly_scores[ood_mask] # prendo soltanto i punteggi degli ood
    ind_out = anomaly_scores[ind_mask] # prendo soltanto i punteggi degli ind

    ood_label = np.ones(len(ood_out)) # creo un array di 1 farà da label
    ind_label = np.zeros(len(ind_out)) # creo un array di 0 farà da label
    
    val_out = np.concatenate((ind_out, ood_out)) # array con tutti i punteggi delle classi ind e out
    val_label = np.concatenate((ind_label, ood_label)) #array con tutte le label delle classi ind e out

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    # Plot ROC and PR curves
    if args.save_plots_dir:
        os.makedirs(args.save_plots_dir, exist_ok=True) # Create the directory if it does not exist
        plot_roc(val_out, val_label, title=f"ROC curve (AUPRC = {prc_auc*100:.2f}%)", save_path=args.save_plots_dir, file_name=f"{args.model}_{args.method}_ROC_curve")
        plot_pr(val_out, val_label, title=f"PR curve (FPR@TPR95 = {fpr*100:.2f}%)", save_path=args.save_plots_dir, file_name=f"{args.model}_{args.method}_PR_curve")
        plot_barcode(val_out, val_label, save_path=args.save_plots_dir, file_name=f"{args.model}_{args.method}_barcode")
    
    # Save the colored score of the first image
    if args.save_colored:
        path = glob.glob(os.path.expanduser(str(args.input)))[0] # First image path
        file_name = osp.splitext(osp.basename(args.save_colored))[0] # File name without extension
        save_dir = osp.dirname(args.save_colored) # Directory where to save the image
        os.makedirs(save_dir, exist_ok=True) # Create the directory if it does not exist
        save_colored_score_image(path, anomaly_score_list[0], save_path=save_dir, file_name=file_name)

        # save also the corresponding label mask
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        save_colored_score_image(pathGT, ood_gts_list[0], save_path=save_dir, file_name="ground_truth")
        
        # Copy also the original image, cropped to the same size
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (anomaly_score_list[0].shape[1], anomaly_score_list[0].shape[0]))
        cv2.imwrite(f"{save_dir}/image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()