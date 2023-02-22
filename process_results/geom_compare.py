
import argparse
import numpy as np
from geom_utils import *
import os
import pickle
import matplotlib.pyplot as plt
from plot_utils import *

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--save_dir', type=str, default='../saved_logs', help='path to experiment directory')
parser.add_argument('--plot_dir', type=str, default='../plot_dir', help='path to experiment directory')
parser.add_argument('--loss_type', default='CDT', type=str, choices=['CDT', 'LDT'], help='Imbalance loss type')
parser.add_argument("--K", type=int, default=10)
parser.add_argument('--R', type=int, default=10, help='Imbalance ratio')
parser.add_argument("--augmentation", action='store_true')
parser.add_argument('--rho', type=float, default=0.5, help='Step imbalance cutoff')
parser.add_argument("--versions", type=int, default=1)
parser.add_argument('--gamma_values', type=float, nargs='+', default = np.linspace(-1.5, 1.5, num = 13).tolist(), help='List of gamma values')
args = parser.parse_args()


        
def main():

    maj_classes = [c for c in range(0, int(args.K * args.rho))]
    min_classes = [c for c in range(0, args.K) if c not in maj_classes]
    
    thm_geom_item = thm_geom()
    thm_geom_item.calc_geom(args, min_classes, maj_classes)

    ufm_general_save_dir = args.save_dir + '/R_' + str(args.R) + "/" + "UFM" + "/" + str(args.loss_type) + "/" 
    ufm_exp_geom_item = exp_geom(scenario = "UFM", gamma_values = args.gamma_values)
    ufm_exp_geom_item.calc_geom(args, ufm_general_save_dir, maj_classes, min_classes, 0)
    
    # mnist_mlp_general_save_dir = args.save_dir + '/R_' + str(args.R) + "/" + '_'.join(["MLP", "MNIST", args.augmentation]) + "/" + str(args.loss_type) + "/" 
    # mnist_mlp_exp_geom_item = exp_geom(scenario = "mnist_MLP", gamma_values = args.gamma_values)
    # mnist_mlp_exp_geom_item.calc_geom(args, mnist_mlp_general_save_dir, maj_classes, min_classes, 0)

    # cifar_resnet_general_save_dir = args.save_dir + '/R_' + str(args.R) + "/" + '_'.join(["ResNet18", "CIFAR10", args.augmentation]) + "/" + str(args.loss_type) + "/" 
    # cifar_resnet_exp_geom_item = exp_geom(scenario = "cifar_resnet", gamma_values = args.gamma_values)
    # cifar_resnet_exp_geom_item.calc_geom(args, cifar_resnet_general_save_dir, maj_classes, min_classes, 0)

    
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    color_theorem = "#3A5FCD"
    color_resnet = "red"
    color_MLP = "green"
    color_UFM = "orange"


    linewidth_theorem_etf = 2
    linewidth_theorem_seli = 5
    size_resnet = 15
    size_MLP = 11.5
    size_UFM = 13.5

    marker_resnet = "^"
    marker_UFM = "o"
    marker_MLP = "s"

    plot_thm = True
    plot_UFM = True
    plot_MLP = False
    plot_resnet = False

    loss_name = "CDT"
    file_name = "Presentation_" + loss_name + "_Thm_vs_Exp_Plot" 
    if plot_thm:
        file_name += "_Thm"
    if plot_UFM:
        file_name += "_UFM"
    if plot_MLP:
        file_name += "_MLP"
    if plot_resnet:
        file_name += "_resnet"
        file_name += "_R" + str(args.R)


    fig, axs = plot_setup()
    
    if plot_thm:
        plot_theorem(thm_geom_item, axs, color_theorem, linewidth_theorem_etf, linewidth_theorem_seli, args.K)

    if plot_UFM:
        plot_exp_scenario(ufm_exp_geom_item, axs, args.gamma_values, color_UFM, size_UFM, marker_UFM, "SGD UFM")

    # if plot_MLP:
    #     plot_scenario(axs, experimental_gamma_values_converged_MLP, color_MLP, size_MLP, marker_MLP, "SGD MLP",
    #                         W_ratio_dict_MLP, W_maj_maj_cos_dict_MLP, W_min_min_cos_dict_MLP, W_maj_min_cos_dict_MLP,
    #                         H_ratio_dict_MLP, H_maj_maj_cos_dict_MLP, H_min_min_cos_dict_MLP, H_maj_min_cos_dict_MLP)

    # if plot_resnet:
    #     plot_scenario(axs, experimental_gamma_values_converged_resnet, color_resnet, size_resnet, marker_resnet, "SGD Resnet",
    #                         W_ratio_dict_resnet, W_maj_maj_cos_dict_resnet, W_min_min_cos_dict_resnet, W_maj_min_cos_dict_resnet,
    #                         H_ratio_dict_resnet, H_maj_maj_cos_dict_resnet, H_min_min_cos_dict_resnet, H_maj_min_cos_dict_resnet)

    legend = axs[1,3].legend(fontsize=20, loc='lower right', bbox_to_anchor=(0.95, 0.05))

    fig.suptitle(loss_name + " Loss", fontweight='bold', fontsize = 30)
    plt.subplots_adjust(wspace=0.25, hspace=0.02)


    plt.savefig(file_name+ ".pdf",bbox_inches='tight', dpi=1200)
    plt.show()


if __name__ == '__main__':
    main()