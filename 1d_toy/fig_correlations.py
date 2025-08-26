# %%
# """
# Function that takes in two parameters, a and b and returns a spiked waveform 
# of the form exp(-ax)*sin(bx)
# """
# function spikedWaveform(x, a, b)
#     return exp.(-a * x) .* sin.(b * x)
# end

# """
# Function that returns Taylor series approximation of the spiked waveform exp(-ax)*sin(bx).
# This acts as our low-fidelity model approximation.
# """
# function taylorApprox(x, a, b)
#     sinTaylor = b*x - (b^3/factorial(3))*x.^3 + (b^5/factorial(5))*x.^5;
#     return exp.(-a*x) .* sinTaylor
# end

# Experiment with different forms of LoFi model and also try changing the range of b.

# https://www.tandfonline.com/doi/epdf/10.4169/math.mag.84.2.098?needAccess=true



import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats

def spiked_waveform(x, a, b):
    """
    Function that takes in two parameters, a and b and returns a spiked waveform
    of the form exp(-ax)*sin(bx).
    """
    return np.exp(-a * x) * np.sin(b * x)

def taylor_approx(x, a, b):
    """
    Function that returns Taylor series approximation of the spiked waveform exp(-ax)*sin(bx).
    This acts as our low-fidelity model approximation.
    """
    sin_taylor = b * x - (b**3 / math.factorial(3)) * x**3 + (b**5 / math.factorial(5)) * x**5
    return np.exp(-a * x) * sin_taylor

def b_sine_approx(x, a, b):
    """
    Function that returns a degraded version of Bhaskara's sine approximation on the interval [0, π].
    """
    bx_d = x * (180 / np.pi) * b
    return np.exp(-a * x) * (3.5 * bx_d * (180 - bx_d)) / (15000 - bx_d * (180 - bx_d))


def calculate_correlation(x, a, b, functions=[spiked_waveform, taylor_approx]):
    f_values = functions[0](x, a, b)
    g_values = functions[1](x, a, b)
    correlation, _ = stats.pearsonr(f_values, g_values)
    return correlation

def main(case='consistent'):
    num_samples = 1000
    num_points = 250
    x = np.linspace(0, 0.1, num_points)
    
    # Generate samples for a and b
    a_samples = np.random.uniform(40, 60, num_samples)
    # b_samples = np.random.uniform(60, 80, num_samples)
    if case == 'consistent':
        b_samples = np.random.uniform(30, 50, num_samples)
    elif case == 'inconsistent':
        b_samples = np.random.uniform(60, 80, num_samples)
    
    # Calculate correlations for each sample
    # correlations = np.zeros(num_samples)
    correlations = np.zeros(num_points)
    for i in range(num_points):
        if case == 'consistent':
            correlations[i] = calculate_correlation(x[i], a_samples, b_samples, functions=[spiked_waveform, b_sine_approx])
        elif case == 'inconsistent':
            correlations[i] = calculate_correlation(x[i], a_samples, b_samples, functions=[spiked_waveform, taylor_approx])

    return correlations
    
    # Calculate mean and standard deviation of correlations
    # mean_correlation = np.mean(correlations)
    # std_correlation = np.std(correlations)
    
    # # Plot the results
    # fig1 = plt.figure(figsize=(10, 6))
    # # plt.errorbar(a_samples, mean_correlation, yerr=std_correlation, fmt='o-', capsize=5, capthick=1)
    # # plt.scatter(a_samples, correlations, s=5)
    # plt.plot(x, correlations, linewidth=3.5)
    # plt.xlabel('x', fontsize=22)
    # plt.ylabel('Correlation', fontsize=22)
    # plt.ylim()
    # plt.title(r'Correlation between $y_{HF}(x, a, b)$ and $y_{LF}(x, a, b)$', fontsize=22)
    # # change size of ticks
    # plt.xticks(fontsize=17)
    # plt.yticks(fontsize=17)
    # plt.grid(True)

    # if saveFig:
    #     if case == 'consistent':
    #         fig1.savefig('./Plots/correlation_plot_1d_toy_modified_consistent.png',
    #             dpi=300)
    #     elif case == 'inconsistent':
    #         fig1.savefig('./Plots/correlation_plot_1d_toy_modified_inconsistent.png',
    #             dpi=300)
    
    # plt.show()

# %%

# main()
corr_consistent = main(case='consistent')
corr_inconsistent = main(case='inconsistent')

# %%
fig, ax = plt.subplots(1, 2, figsize=(18, 7), sharex=True)

x = np.linspace(0, 0.1, 250)

ax[0].plot(x, corr_inconsistent, color='blue', linewidth=3.5)
ax[0].set_xlabel(r"$x$", fontsize=20)
ax[0].set_ylabel('Correlation', fontsize=20)
ax[0].set_title("C1", fontsize=24)
ax[0].tick_params(axis='both', which='major', labelsize=20)
ax[0].grid(True)
ax[0].set_xlim([0, 0.1])
ax[1].plot(x, corr_consistent, color='blue', linewidth=3.5)
ax[1].set_xlabel(r"$x$", fontsize=20)
# ax[1].set_ylabel('Correlation', fontsize=20)
ax[1].set_title("C2", fontsize=24)
ax[1].tick_params(axis='both', which='major', labelsize=20)
ax[1].grid(True)
ax[1].set_xlim([0, 0.1])
# plt.subplots_adjust(wspace=0.5)

plt.savefig("/Users/ajivani/Desktop/Research/KLE_UQ_Final/1d_toy/figs_1d/correlation_plot_1d_toy_modified_C1_C2.png", bbox_inches='tight')





# %%
