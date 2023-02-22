import numpy as np
import os
import pickle
from loggers import *

def CDT_geom(R, delta_maj, delta_min, k):
    
    # for rho = 0.5
    w_norm = {}
    w_cosine = {}

    temp_min = np.sqrt((delta_min ** (-2)) + (delta_maj ** (-2))) * delta_min
    temp_maj = np.sqrt((delta_min ** (-2)) + (delta_maj ** (-2))) * delta_maj



    w_norm['maj'] = ((np.sqrt(R) * (k-2) + 2 * np.sqrt(R+1) * (temp_min ** (-3))) / 
                    k / delta_maj)
    w_norm['min'] = ((1 * (k-2) + 2 * np.sqrt(R+1) * (temp_maj ** (-3))) /
            k / delta_min)
    w_norm['maj-min'] = w_norm['maj'] / w_norm['min']

    w_cosine['maj'] = ((-2*np.sqrt(R)    + 2*np.sqrt(R+1)*(temp_min**(-3))) / 
                    ((k-2)*np.sqrt(R) + 2*np.sqrt(R+1)*(temp_min**(-3))))

    w_cosine['min'] = ((-2     + 2*np.sqrt(R+1)*(temp_maj**(-3))) / 
                    ((k-2) + 2*np.sqrt(R+1)*(temp_maj**(-3))))

    w_cosine['maj-min'] = -2 * ((np.sqrt(R+1) * ((delta_min*delta_maj)**(-2))) / 
                            (k*np.sqrt(w_norm['maj'] * w_norm['min']) * (np.sqrt((delta_min ** (-2)) + (delta_maj ** (-2))) ** 3)))


    h_norm = {}
    h_cosine = {}

    temp = np.sqrt(((delta_min ** (-2)) + (delta_maj ** (-2))) * (R+1))

    h_norm['maj'] = ((k-2)/np.sqrt(R) + 2 *  (delta_min ** (-1)) / temp) / k / delta_maj
    h_norm['min'] = ((k-2) + 2 *  (delta_maj ** (-1)) / temp) / k / delta_min 
    h_norm['maj-min'] = h_norm['maj'] / h_norm['min']

    h_cosine['maj'] = ((-2/np.sqrt(R) + 2 *  (delta_min ** (-1)) / temp) / 
                        ((k-2)/np.sqrt(R) + 2 *  (delta_min ** (-1)) / temp))
    h_cosine['min'] = ((-2 + 2 *  (delta_maj ** (-1)) / temp) / 
                        ((k-2) + 2 *  (delta_maj ** (-1)) / temp))
    h_cosine['maj-min'] = -((2 * (delta_min ** (-1)) * (delta_maj ** (-1)) / temp) /
                            (k * np.sqrt(h_norm['maj'] * h_norm['min'])))

    wh_cosine = {}
    wh_cosine['maj'] = ((k * (delta_min**(-2)) + (k-2) * (delta_maj**(-2))) / 
                        (k * delta_maj * ((delta_min**(-2)) + (delta_maj**(-2))) * np.sqrt(w_norm['maj'] * h_norm['maj'])))
    wh_cosine['min'] = ((k * (delta_maj**(-2)) + (k-2) * (delta_min**(-2))) / 
                        (k * delta_min * ((delta_min**(-2)) + (delta_maj**(-2))) * np.sqrt(w_norm['min'] * h_norm['min'])))

    return w_norm, w_cosine, h_norm, h_cosine, wh_cosine



def LDT_geom(R, delta_maj, delta_min, k):
    
    # for rho = 0.5
    w_norm = {}
    w_cosine = {}
    h_norm = {}
    h_cosine = {}
    wh_cosine = {}

    w_norm['maj']     = np.sqrt(R) * (1- 2/k) + np.sqrt((R+1)/2)/k
    w_norm['min']     =              (1- 2/k) + np.sqrt((R+1)/2)/k
    w_norm['maj-min'] = w_norm['maj'] / w_norm['min']

    w_cosine['maj']     = ((-2 * np.sqrt(R) + np.sqrt((R+1)/2)) /
                            ((k-2)*np.sqrt(R) + np.sqrt((R+1)/2)))
    w_cosine['min']     = ((R - 7)/
                            (R - 7 + 2*k*(2 + np.sqrt((R+1)/2))))
    w_cosine['maj-min'] = -((np.sqrt((R+1)/2)) / 
                            (k * np.sqrt(w_norm['maj'] * w_norm['min'])))

    h_norm['maj']     = (1 - 2/k)/np.sqrt(R) + 1/(k * np.sqrt((R+1)/2))
    h_norm['min']     = (1 - 2/k)            + 1/(k * np.sqrt((R+1)/2))
    h_norm['maj-min'] = h_norm['maj'] / h_norm['min']

    h_cosine['maj']     = -(R+2) / (-(R+2) + k*(R+1 + np.sqrt(R*(R+1)/2)))
    h_cosine['min']     = ((1 - np.sqrt(2*(R+1))) /
                            (1 - np.sqrt(2*(R+1)) + k*np.sqrt((R+1)/2)))
    h_cosine['maj-min'] = -1 / (k * np.sqrt((R+1)/2) * np.sqrt(h_norm['maj'] * h_norm['min']))

    wh_cosine['maj'] = (1-1/k) / np.sqrt(w_norm['maj']*h_norm['maj'])
    wh_cosine['min'] = (1-1/k) / np.sqrt(w_norm['min']*h_norm['min'])


    h_norm['maj'] = h_norm['maj'] / (delta_maj ** 2) 
    h_norm['min'] = h_norm['min'] / (delta_min ** 2) 
    h_norm['maj-min'] = h_norm['maj'] / h_norm['min']

    return w_norm, w_cosine, h_norm, h_cosine, wh_cosine#, norm_true, w_cosine_true


def normalize_deltas(delta_list, gamma):
    Delta_list = np.array(delta_list) ** gamma
    Delta_list = len(Delta_list) * Delta_list / sum(Delta_list)
    return Delta_list



class thm_geom():
    def __init__(self):
        self.w_norm = {'maj':[],
            'min':[],
            'maj-min': []}

        self.w_cosine = {'maj': [],
                    'min': [],
                    'maj-min': []}
        self.h_norm = {'maj':[],
                'min':[],
                'maj-min': []}

        self.h_cosine = {'maj': [],
                    'min': [],
                    'maj-min': []}
        self.wh_cosine = {'maj': [],
                        'min': []}

        self.gamma_vector = np.arange(-2,2,0.05)

    def calc_geom(self, args, min_classes, maj_classes):
        delta_list = [args.R if c in maj_classes else 1 for c in range(0, args.K)]

        for gamma_value in self.gamma_vector:

            delta_dict_for_gamma = normalize_deltas(delta_list ,gamma_value)

            if args.loss_type == "CDT":
                w_norm_, w_cosine_, h_norm_, h_cosine_, wh_cosine_ = CDT_geom(args.R, delta_dict_for_gamma[maj_classes[0]], delta_dict_for_gamma[min_classes[-1]], args.K)
            if args.loss_type == "LDT":
                w_norm_, w_cosine_, h_norm_, h_cosine_, wh_cosine_ = LDT_geom(args.R, delta_dict_for_gamma[maj_classes[0]], delta_dict_for_gamma[min_classes[-1]], args.K)

            for block in ['maj', 'min', 'maj-min']:
                self.w_norm[block].append(w_norm_[block])
                self.h_norm[block].append(h_norm_[block])

                self.w_cosine[block].append(w_cosine_[block])
                self.h_cosine[block].append(h_cosine_[block])

                if block != 'maj-min':
                    self.wh_cosine[block].append(wh_cosine_[block])
    
        return


class exp_geom():
    def __init__(self, scenario, gamma_values):
        self.scenario = scenario
        self.gamma_values = gamma_values
        self.experimental_gamma_values_converged = []
        self.W_ratio_dict = {}
        self.W_maj_maj_cos_dict = {}
        self.W_min_min_cos_dict = {}
        self.W_maj_min_cos_dict = {}
        self.H_ratio_dict = {}
        self.H_maj_maj_cos_dict = {}
        self.H_min_min_cos_dict = {}
        self.H_maj_min_cos_dict = {}
        self.W_H_maj_cos_dict = {}
        self.W_H_min_cos_dict = {}

        self.accuracies = {}
        self.accuracies_maj = {}
        self.accuracies_min = {}
    
    def calc_geom(self, args, general_save_dir, maj_classes, min_classes, version):

        for gamma in self.gamma_values:
            file_path = general_save_dir + '_'.join(['gamma', str(gamma), "ver", str(version)]) + "/logger_train"

            if os.path.exists(file_path):
                file = open(file_path, 'rb')
                logger = pickle.load(file)

                maj_train_accuracies = []
                min_train_accuracies = []
                for i in range(0, len(logger.acc_perclass)):
                    maj_train_accuracies.append(sum([logger.acc_perclass[i][c] for c in maj_classes]) / 5)
                    min_train_accuracies.append(sum([logger.acc_perclass[i][c] for c in min_classes]) / 5)
                self.accuracies_maj[gamma] = max(maj_train_accuracies[-10:])
                self.accuracies_min[gamma] = max(min_train_accuracies[-10:])
                self.accuracies[gamma] = max(logger.accuracy[-10:])

                Last_W = logger.W[-1]
                Last_W = Last_W / np.linalg.norm(Last_W, 'fro')
                
                Last_H = logger.M[-1]
                Last_H = Last_H / np.linalg.norm(Last_H, 'fro')

                
                train_accuracy = sum(logger.acc_perclass[-1].values()) / args.K
                if train_accuracy >= 0.99:
                    self.experimental_gamma_values_converged.append(gamma)
                
                ############################################################################ ############################################################################
                
                ######################## W_maj_min_norm_ratio_dict
                sum_ratios  = 0
                for i in maj_classes:
                    for j in min_classes:
                        sum_ratios += np.linalg.norm(Last_W[i,:], ord = 2)**2 / np.linalg.norm(Last_W[j,:], ord = 2)**2
                self.W_ratio_dict[gamma] = sum_ratios / 25

                ######################## W_maj_maj_cos_dict
                W_maj_cos_ave = 0
                for i in maj_classes:
                    for j in maj_classes:
                        if i != j:
                            W_maj_cos_ave += np.dot(Last_W[i,:], Last_W[j,:]) / (np.linalg.norm(Last_W[i,:], ord = 2) * np.linalg.norm(Last_W[j,:], ord = 2))
                self.W_maj_maj_cos_dict[gamma] = W_maj_cos_ave / (2 * args.K)

                ######################## W_min_min_cos_dict
                W_min_cos_ave = 0
                for i in min_classes:
                    for j in min_classes:
                        if i != j:
                            W_min_cos_ave += np.dot(Last_W[i,:], Last_W[j,:]) / (np.linalg.norm(Last_W[i,:], ord = 2) * np.linalg.norm(Last_W[j,:], ord = 2))
                self.W_min_min_cos_dict[gamma] = W_min_cos_ave / (2 * args.K)

                ######################## W_maj_min_cos_dict
                W_maj_min_cos_ave = 0
                for i in maj_classes:
                    for j in min_classes:
                        W_maj_min_cos_ave += np.dot(Last_W[i,:], Last_W[j,:]) / (np.linalg.norm(Last_W[i,:], ord = 2) * np.linalg.norm(Last_W[j,:], ord = 2))
                self.W_maj_min_cos_dict[gamma] = W_maj_min_cos_ave / (2 * args.K + args.K // 2)

                ############################################################################ ############################################################################

                ######################## H_maj_min_norm_ratio_dict
                sum_ratios  = 0
                for i in maj_classes:
                    for j in min_classes:
                        sum_ratios += np.linalg.norm(Last_H[:,i], ord = 2)**2 / np.linalg.norm(Last_H[:,j], ord = 2)**2
                self.H_ratio_dict[gamma] = sum_ratios / (2 * args.K + args.K // 2)

                ######################## H_maj_maj_cos_dict
                H_maj_cos_ave = 0
                for i in maj_classes:
                    for j in maj_classes:
                        if i != j:
                            H_maj_cos_ave += np.dot(Last_H[:,i], Last_H[:,j]) / (np.linalg.norm(Last_H[:,i], ord = 2) * np.linalg.norm(Last_H[:,j], ord = 2))
                self.H_maj_maj_cos_dict[gamma] = H_maj_cos_ave / (2 * args.K)

                ######################## H_min_min_cos_dict
                H_min_cos_ave = 0
                for i in min_classes:
                    for j in min_classes:
                        if i != j:
                            H_min_cos_ave += np.dot(Last_H[:,i], Last_H[:,j]) / (np.linalg.norm(Last_H[:,i], ord = 2) * np.linalg.norm(Last_H[:,j], ord = 2))
                self.H_min_min_cos_dict[gamma] = H_min_cos_ave/ (2 * args.K)

                ######################## H_maj_min_cos_dict
                H_maj_min_cos_ave = 0
                for i in maj_classes:
                    for j in min_classes:
                        if i != j:
                            H_maj_min_cos_ave += np.dot(Last_H[:,i], Last_H[:,j]) / (np.linalg.norm(Last_H[:,i], ord = 2) * np.linalg.norm(Last_H[:,j], ord = 2))
                self.H_maj_min_cos_dict[gamma] = H_maj_min_cos_ave / (2 * args.K + args.K // 2)

                ############################################################################ ############################################################################

                ######################## W_H_maj_cos_dict
                W_H_maj_cos_ave = 0
                for i in maj_classes:
                    W_H_maj_cos_ave += np.dot(Last_W[i,:], Last_H[:,i]) / (np.linalg.norm(Last_W[i,:], ord = 2) * np.linalg.norm(Last_H[:,i], ord = 2))
                self.W_H_maj_cos_dict[gamma] = W_H_maj_cos_ave / (args.K // 2)

                ######################## W_H_min_cos_dict
                W_H_min_cos_ave = 0
                for i in min_classes:
                    W_H_min_cos_ave += np.dot(Last_W[i,:], Last_H[:,i]) / (np.linalg.norm(Last_W[i,:], ord = 2) * np.linalg.norm(Last_H[:,i], ord = 2))
                self.W_H_min_cos_dict[gamma] = W_H_min_cos_ave / (args.K // 2)
            
            else:
                continue
    
        return