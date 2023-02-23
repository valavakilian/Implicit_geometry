
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
import pickle

import os
import pickle
import shutil

from generate_cifar import IMBALANCECIFAR10
from generate_mnist import IMBALANCEMNIST

import argparse

from models import *

from criterion import *

from loggers import *

from utils import *

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_dir', type=str, default='../data', help='path to dataset directory')
parser.add_argument('--save_dir', type=str, default='../saved_logs', help='path to experiment directory')
parser.add_argument("--K", type=int, default=10)
parser.add_argument('--R', type=int, default=10, help='Imbalance ratio')
parser.add_argument('--n_min', type=int, default=5, help='Number of datapoints for minority classes')
parser.add_argument('--rho', type=float, default=0.5, help='Step imbalance cutoff')
parser.add_argument("--network_dim", type=int, default=20, help='Inner layer dimension of UFM network')
parser.add_argument("--versions", type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=6000, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument("--gpu", action='store_true')
parser.add_argument('--loss_type', default='CDT', type=str, choices=['wCE', 'CDT', 'LDT'], help='Imbalance loss type')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--lr_decay_epochs', type=float, nargs='+', default = [2000, 4000], help='learning rate decay epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--gamma_values', type=float, nargs='+', default = np.linspace(-1.5, 1.5, num = 13).tolist(), help='List of gamma values')
parser.add_argument("--args_rand", type=int, default=1)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


log_epoch_list = np.hstack((np.arange(0, 20, 2, dtype=np.int32),
                 np.arange(20, 200, 20, dtype=np.int32),
                 np.arange(200, 6000, 50, dtype=np.int32)))



def main():

    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")

    print(vars(args))
    classes = [c for c in range(0, args.K)]
    maj_classes = [c for c in range(0, int(args.K * args.rho))]
    min_classes = [c for c in range(0, args.K) if c not in maj_classes]
    delta_list = [args.R if c in maj_classes else 1 for c in range(0, args.K)]
    n_c_train = [args.n_min * args.R if c in maj_classes else args.n_min for c in range(0, args.K)]

    general_save_dir = args.save_dir + '/R_' + str(args.R) + "/" + "UFM" + "/" + str(args.loss_type) + "/" 
    if not os.path.exists(general_save_dir):
        os.makedirs(general_save_dir, exist_ok=True)
    

    # ------- Imbalanced dataset --------------------------------------------------------------------------------------------------
    N = sum(n_c_train)
    n_min = args.n_min
    n_maj = args.R * n_min
    k_maj = int(args.rho * args.K)
    k_min = args.K - k_maj
    X = torch.eye(N, device=device)     # basis vectors

    y_list = []; cls_num_list = [];

    cls_num_list = np.array([n_maj, n_min])
    cls_num_list = np.repeat(cls_num_list, [k_maj, k_min])

    y_list = np.array([i for i in range(args.K)])
    y_list = np.repeat(y_list, cls_num_list)

    Y = torch.tensor(y_list, device=device).long() 

    # We set the batch size to N in order to have full batch experiments
    args.batch_size = N

    # first index of each class
    temp = [0] + cls_num_list

    dataset = torch.utils.data.TensorDataset(X,Y)        
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    analysis_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ------- Main Training -------------------------------------------------------------------------------------------------------
    for version in range(0, args.versions):
        for gamma in args.gamma_values:
            print("-" * 30)
            print("Performing experiment for " + str('_'.join(['gamma', str(gamma), "ver", str(version)])))

            exp_save_path = general_save_dir + '_'.join(['gamma', str(gamma), "ver", str(version)]) + "/"
            exp_complete_flag = exp_save_path + "exp_complete.txt"

            if not os.path.exists(exp_save_path):
                os.makedirs(exp_save_path, exist_ok=True)
            if not os.path.exists(exp_complete_flag):
                shutil.rmtree(exp_save_path)
                os.makedirs(exp_save_path, exist_ok=True)
            else:
                print("Skipping this experiments since flag is set. Please remove flag to rerun this experiment.")
                continue

            # ------- Model -------------------------------------------------------------------------------------------------------
            # Hook for this experiment's features
            class features:
                pass
            def hook(self, input, output):
                features.value = input[0].clone()
            
            model = TwoLayerLinear(N, args.network_dim, args.K, bias=False)
            classifier = model.fc2
            classifier.register_forward_hook(hook)

            print(model)

            # ------- Loss ---------------------------------------------------------------------------------------------------------
            if args.loss_type == 'wCE':
                wCE_weight = [1 if c in maj_classes else args.R**gamma for c in range(0, args.K)]
                wCE_weight = torch.tensor(wCE_weight)
                wCE_weight = len(wCE_weight) * wCE_weight / sum(wCE_weight)
                criterion = nn.CrossEntropyLoss()
                criterion_summed = nn.CrossEntropyLoss(reduction='sum')
            if args.loss_type == "CDT":
                criterion = CDTLoss(delta_list, gamma=gamma, weight=None, reduction=None, device=device)
                criterion_summed = CDTLoss(delta_list, gamma=gamma, weight=None, reduction="sum", device=device)
            if args.loss_type == "LDT":
                criterion = LDTLoss(delta_list, gamma=gamma, weight=None, reduction=None, device=device)
                criterion_summed = LDTLoss(delta_list, gamma=gamma, weight=None, reduction="sum", device=device)
            
            criterion.to(device)
            criterion_summed.to(device)

            # ------- Optimizer ----------------------------------------------------------------------------------------------------
            optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_decay_epochs,
                                                        gamma=args.lr_decay)
            
            # ------- Data Loggers --------------------------------------------------------------------------------------------------
            logger_train = logger()

            # ------- Train ---------------------------------------------------------------------------------------------------------
            cur_epochs = []
            for epoch in range(1, args.epochs + 1):
                print("print epoch")
                torch.cuda.empty_cache()

                train(model, criterion, args, device, train_loader, optimizer, epoch)
                lr_scheduler.step()
                
                if epoch in log_epoch_list:
                    cur_epochs.append(epoch)
                    torch.cuda.empty_cache()
                    analysis(logger_train, model, criterion_summed, args, device, analysis_loader, classifier, features, epoch)

                    save_logger(logger_train, exp_save_path, "logger_train")

            os.makedirs(exp_complete_flag, exist_ok=True)


if __name__ == '__main__':
    main()