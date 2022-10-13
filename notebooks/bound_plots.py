import numpy as np
from matplotlib import pyplot as plt
from pactl.bounds.get_pac_bounds import pac_bayes_bound_opt
from pactl.bounds.get_pac_bounds import compute_mcallester_bound
from pactl.bounds.get_pac_bounds import compute_convexity_bound
from pactl.bounds.get_pac_bounds import compute_catoni_bound

train_accuracy = 0.710693
train_error = 1. - train_accuracy
divergence = 1724
sample_size = int(3 * 1.e4)

div_linspace = np.linspace(0, divergence, num=100)
bound_c = np.zeros(len(div_linspace))
bound_m = np.zeros(len(div_linspace))
bound_t = np.zeros(len(div_linspace))
bound_cc = np.zeros(len(div_linspace))
for i in range(len(div_linspace)):
    bound_c[i] = pac_bayes_bound_opt(div_linspace[i], train_error, sample_size)
    bound_m[i] = compute_mcallester_bound(train_error, div_linspace[i], sample_size)
    bound_t[i] = compute_convexity_bound(train_error, div_linspace[i], sample_size)
    bound_cc[i] = compute_catoni_bound(train_error, div_linspace[i], sample_size)

plt.style.use('ggplot')
plt.figure(dpi=250)
plt.plot(div_linspace, bound_c, label='Catoni')
plt.plot(div_linspace, bound_m, label='McAllester')
plt.plot(div_linspace, bound_t, label='Convexity')
plt.plot(div_linspace, bound_cc, label='Catoni no geom')
plt.axhline(y=train_error, label=f'Train error {train_error:2.2f}', color='blue')
plt.legend()
plt.show()


def bound_div(div, alpha):
    bound = pac_bayes_bound_opt(div, train_error, sample_size, alpha)
    return bound


div_linspace = np.linspace(0, 1724, num=100)
alphas = [1.e-4, 1.e-1, 1.e-0, 1.e-8]
bounds = np.zeros(len(div_linspace))
bounds2 = np.zeros(len(div_linspace))
bounds3 = np.zeros(len(div_linspace))
bounds4 = np.zeros(len(div_linspace))
for i in range(len(div_linspace)):
    bounds[i] = bound_div(div_linspace[i], alpha=alphas[0])
    bounds2[i] = bound_div(div_linspace[i], alpha=alphas[1])
    bounds3[i] = bound_div(div_linspace[i], alpha=alphas[2])
    bounds4[i] = bound_div(div_linspace[i], alpha=alphas[3])


plt.style.use('ggplot')
plt.figure(dpi=250)
plt.plot(div_linspace, bounds3, label=f'alpha = {alphas[2]}')
plt.plot(div_linspace, bounds, label=f'alpha = {alphas[0]}')
plt.plot(div_linspace, bounds2, label=f'alpha = {alphas[1]}')
plt.plot(div_linspace, bounds4, label=f'alpha = {alphas[3]}')
plt.axhline(y=train_error, label=f'Train error {train_error:2.2f}', color='blue')
plt.legend()
plt.show()

# train_accuracies = [0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
train_accuracies = [0.70, 0.8, 0.9, 0.95, 0.98]

plt.style.use('ggplot')
plt.figure(dpi=250)
for train_accuracy in train_accuracies:
    for i in range(len(div_linspace)):
        train_error = 1. - train_accuracy
        bounds[i] = pac_bayes_bound_opt(div_linspace[i], train_error, n=sample_size)
    plt.plot(div_linspace, bounds - train_error, label=f'Train Acc {train_accuracy}')
plt.legend()
plt.ylabel('Estimated Generalization Gap')
plt.xlabel('Divergence')
plt.show()
