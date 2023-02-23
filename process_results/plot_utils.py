import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
sns.set_context("paper")
sns.set_style("whitegrid")      

def plot_exp_scenario(exp_geom_item, axs, gamma_values, color, size, marker, label):
    
    for gamma_value in gamma_values:
      axs[0,0].plot(gamma_value, exp_geom_item.W_ratio_dict[gamma_value], c = color, marker=marker, ms=size)
      axs[0,1].plot(gamma_value, exp_geom_item.W_maj_maj_cos_dict[gamma_value], c = color, marker=marker, ms=size)
      axs[0,2].plot(gamma_value, exp_geom_item.W_min_min_cos_dict[gamma_value], c = color, marker=marker, ms=size)
      axs[0,3].plot(gamma_value, exp_geom_item.W_maj_min_cos_dict[gamma_value], c = color, marker=marker, ms=size)
  
    for gamma_value in gamma_values:
      axs[1,0].plot(gamma_value, exp_geom_item.H_ratio_dict[gamma_value], c = color, marker=marker, ms=size)
      axs[1,1].plot(gamma_value, exp_geom_item.H_maj_maj_cos_dict[gamma_value], c = color, marker=marker, ms=size)
      axs[1,2].plot(gamma_value, exp_geom_item.H_min_min_cos_dict[gamma_value], c = color, marker=marker, ms=size)
    for gamma_value in gamma_values[:-5]:
      axs[1,3].plot(gamma_value, exp_geom_item.H_maj_min_cos_dict[gamma_value], c = color, marker=marker, ms=size)
    axs[1,3].plot(gamma_values[-5], exp_geom_item.H_maj_min_cos_dict[gamma_values[-5]], c = color, marker=marker, ms=size, label = label)
    for gamma_value in gamma_values[-4:]:
      axs[1,3].plot(gamma_value, exp_geom_item.H_maj_min_cos_dict[gamma_value], c = color, marker=marker, ms=size)
    
    return

def plot_theorem(thm_geom_item, axs, color_theorem, linewidth_theorem_etf, linewidth_theorem_seli, K):

    gamma_vector = thm_geom_item.gamma_vector

    axs[0,0].plot(gamma_vector, np.ones(gamma_vector.shape), '--', c=color_theorem, linewidth=linewidth_theorem_etf)
    axs[0,1].plot(gamma_vector, -np.ones(gamma_vector.shape)/(K-1), '--', c=color_theorem, linewidth=linewidth_theorem_etf)
    axs[0,2].plot(gamma_vector, -np.ones(gamma_vector.shape)/(K-1), '--', c=color_theorem, linewidth=linewidth_theorem_etf)
    axs[0,3].plot(gamma_vector, -np.ones(gamma_vector.shape)/(K-1), '--', c=color_theorem, linewidth=linewidth_theorem_etf)

    axs[0,0].plot(gamma_vector, thm_geom_item.w_norm['maj-min'], c=color_theorem, 
                linewidth=linewidth_theorem_seli, label='k={}'.format(K))
    axs[0,1].plot(gamma_vector, thm_geom_item.w_cosine['maj'], c=color_theorem, 
                linewidth=linewidth_theorem_seli, label='k={}'.format(K))
    axs[0,2].plot(gamma_vector, thm_geom_item.w_cosine['min'], c=color_theorem, 
                linewidth=linewidth_theorem_seli, label='k={}'.format(K))
    axs[0,3].plot(gamma_vector, thm_geom_item.w_cosine['maj-min'], c=color_theorem, 
              linewidth=linewidth_theorem_seli, label='k={}'.format(K))
    
    axs[1,0].plot(gamma_vector, np.ones(gamma_vector.shape), '--', c=color_theorem, linewidth=linewidth_theorem_etf)
    axs[1,1].plot(gamma_vector, -np.ones(gamma_vector.shape)/(K-1), '--', c=color_theorem, linewidth=linewidth_theorem_etf)
    axs[1,2].plot(gamma_vector, -np.ones(gamma_vector.shape)/(K-1), '--', c=color_theorem, linewidth=linewidth_theorem_etf)
    axs[1,3].plot(gamma_vector, -np.ones(gamma_vector.shape)/(K-1), '--', c=color_theorem, linewidth=linewidth_theorem_etf, label = "ETF")

    axs[1,0].plot(gamma_vector, thm_geom_item.h_norm['maj-min'], c=color_theorem, 
                linewidth=linewidth_theorem_seli, label='k={}'.format(K))
    axs[1,1].plot(gamma_vector, thm_geom_item.h_cosine['maj'], c=color_theorem, 
                linewidth=linewidth_theorem_seli, label='k={}'.format(K))
    axs[1,2].plot(gamma_vector, thm_geom_item.h_cosine['min'], c=color_theorem, 
                linewidth=linewidth_theorem_seli, label='k={}'.format(K))
    axs[1,3].plot(gamma_vector, thm_geom_item.h_cosine['maj-min'], c=color_theorem, 
              linewidth=linewidth_theorem_seli, label = "Implicit Geometry")
    
    return


def plot_setup():

    fig, axs = plt.subplots(2, 4, figsize=(4 * 10, 2 * 6))

    for i in range(4):
      # axs[0,i].set_xscale('log')
      # axs[0,i].set_yscale('log')
      axs[0,i].grid(True)
      axs[0,i].tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False) # labels along the bottom edge are off


    axs[0,0].set_ylim([-1.0,20.0])
    axs[0,1].set_ylim([-1.1,0.5])
    axs[0,2].set_ylim([-1.1,1.0])
    axs[0,3].set_ylim([-1.1,0.5])
    axs[0,0].set_title('Norm Ratios', fontweight='bold', fontsize=30)
    axs[0,1].set_title('Maj Angles', fontweight='bold', fontsize=30)
    axs[0,2].set_title('Min Angles', fontweight='bold', fontsize=30)
    axs[0,3].set_title('Maj/Min Angles', fontweight='bold', fontsize=30)
    axs[0,0].set_ylabel('$\mathbf{||w_{maj}||^2/||w_{min}||^2}$', fontweight='bold', fontsize=30)
    axs[0,1].set_ylabel('$\mathbf{cos(w_{maj},w_{maj})}$', fontweight='bold', fontsize=30)
    axs[0,2].set_ylabel('$\mathbf{cos(w_{min},w_{min})}$', fontweight='bold', fontsize=30)
    axs[0,3].set_ylabel('$\mathbf{cos(w_{maj},w_{min})}$', fontweight='bold', fontsize=30)

    for i in range(4):
      axs[1,i].grid(True)
      axs[1,i].set_xlabel('$\mathbf{\gamma}$', fontsize=30)


    axs[1,0].set_ylim([-0.5,9.0])
    axs[1,1].set_ylim([-1.05,1.8])
    axs[1,2].set_ylim([-1.05,0.5])
    axs[1,3].set_ylim([-1.05,0.2])
    axs[1,0].set_ylabel('$\mathbf{||h_{maj}||^2/||h_{min}||^2}$', fontweight='bold', fontsize=30)
    axs[1,1].set_ylabel('$\mathbf{cos(h_{maj},h_{maj})}$', fontweight='bold', fontsize=30)
    axs[1,2].set_ylabel('$\mathbf{cos(h_{min},h_{min})}$', fontweight='bold', fontsize=30)
    axs[1,3].set_ylabel('$\mathbf{cos(h_{maj},h_{min})}$', fontweight='bold', fontsize=30)

    return fig, axs