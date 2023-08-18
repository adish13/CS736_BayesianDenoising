import numpy as np
import matplotlib.pyplot as plt
from numpy import roll as r

# function for calculating prior
def calc_prior(x, function, gamma):
    p1 = function(x - r(x, 1, axis=0), gamma)
    p2 = function(x - r(x, -1, axis=0), gamma)
    p3 = function(x - r(x, 1, axis=1), gamma)
    p4 = function(x - r(x, -1, axis=1), gamma)
    return p1+p2+p3+p4

def calc_prior_grad(x, function, gamma):
    g1 = function(x - r(x, 1, axis=0), gamma)
    g2 = function(x - r(x, -1, axis=0), gamma)
    g3 = function(x - r(x, 1, axis=1), gamma)
    g4 = function(x - r(x, -1, axis=1), gamma)
    return g1+g2+g3+g4

# calculate RRMSE
def RRMSE(img1, img2):
    num = np.sqrt(np.sum((abs(img1) - abs(img2))**2))
    denominator = np.sqrt(np.sum(img1**2))
    ratio = num/denominator
    return ratio

# quadratic prior is independent of gamma
def quadratic_function(u, gamma):
    value = np.sum(np.abs(u)**2)
    return value

def quadratic_gradient(u,gamma):
    gradient = 2*u
    return gradient

# defining huber function
def huber_function(u, gamma):
    condition1 = np.abs(u) <= gamma
    condition2 = np.abs(u) > gamma
    value = np.sum(0.5*(np.abs(u)**2)*condition1 +
                   (gamma*np.abs(u) - 0.5*gamma**2)*condition2)
    return value

def huber_gradient(u, gamma):
    condition1 = np.abs(u) <= gamma
    condition2 = np.abs(u) > gamma
    gradient = u*(condition1) + gamma*np.sign(u)*(condition2)
    return gradient

# defining adaptive function
def adaptive_function(u, gamma):
    value = np.sum(gamma*np.abs(u) - (gamma**2)*np.log(1+np.abs(u)/gamma))
    return value

def adaptive_gradient(u, gamma):
    gradient = u*gamma/(gamma+np.abs(u))
    return gradient

# defining likelihood function for gaussian
def likelihood_function(x, y):
    value = np.sum(np.abs(x-y)**2)
    return value

def likelihood_grad(x, y):
    gradient = 2*(x-y)
    return gradient

def calc_posterior(x, y, prior_func, alpha, gamma=None):
    prior = calc_prior(x, prior_func, gamma)
    likelihood = likelihood_function(x, y)
    posterior = alpha*prior + (1-alpha)*likelihood
    return posterior

def calc_posterior_grad(x, y, grad_func, alpha, gamma=None):
    prior_gradient = calc_prior_grad(x, grad_func, gamma)
    likelihood_gradient = likelihood_grad(x, y)
    posterior_gradient = alpha*prior_gradient + (1-alpha)*likelihood_gradient

    return posterior_gradient

dict = {'quadratic': quadratic_function,'huber': huber_function, 'adaptive': adaptive_function, }
grad_dict = {'quadratic': quadratic_gradient,'huber': huber_gradient, 'adaptive': adaptive_gradient, }

def plot_images(x,noisy_img,noiseless_img,lp_values,runs,prior,dir):
    print('Initial RRMSE between Noiseless and Noisy Image:', round(RRMSE(noiseless_img, noisy_img), 7))
    print('Image denoised in {} runs. RRMSE between Noiseless and denoised image {:.7f}'.format(
            runs, RRMSE(noiseless_img, x)))

    # clear plot
    plt.clf()
    plt.plot(np.arange(runs+1), lp_values)

    # X axis represents number of iterations
    plt.xlabel('No. of iterations')

    # Y axis displays values of objective functions
    plt.ylabel('Objective Function (log)')

    plt.title(prior + ' prior')
    plt.savefig(dir+'/'+prior+'_objective_function.png')
    plt.clf()

    # Required to use "jet" color map
    cmap = 'jet'

    # Saving the denoised image obtained from algorithm
    plt.imshow(x, cmap=cmap)
    plt.title('{}_prior_denoised_img'.format(prior))
    plt.savefig(dir + '/'+prior+'_prior_denoised_img.png')
    plt.clf()

    # Saving the given noisy image
    plt.imshow(noisy_img, cmap=cmap)
    plt.title('Noisy Image')
    plt.savefig(dir+'/noisy_img.png')
    plt.clf()

    # Saving the noiseless image
    plt.imshow(noiseless_img, cmap=cmap)
    plt.title('True image')
    plt.savefig(dir+'/true_image.png')
    plt.clf()

def ImageDenoise(noisy_img, noiseless_img, alpha=0.1, gamma=1, stepsize=1e-2, optimise=False, prior='quadratic', dir='../results/q1'):

    prior_func = dict[prior]
    grad_func = grad_dict[prior]

    if prior_func == 'quadratic_function':
        gamma = None
    else:
        gamma = gamma

    x = noisy_img
    y = noisy_img
    step_size = stepsize

    # lp shorthand for log posterior
    initial_lp = calc_posterior(x, y, prior_func, alpha, gamma)

    runs = 0
    lp_values = [initial_lp]

    # setting variables for number of max runs and least step size
    max_runs = 200
    least_step_size = 1e-8

    while step_size > least_step_size and runs < max_runs:

        # gradient descent
        lp = calc_posterior(x, y, prior_func, alpha, gamma)
        lp_grad = calc_posterior_grad(x, y, grad_func, alpha, gamma)

        x = x - step_size*lp_grad

        lp = calc_posterior(x, y, prior_func, alpha, gamma)
        lp_values.append(lp)

        lr_increase_factor = 1.1
        lr_decrease_factor = 0.5

        # Variable Learning Rate
        if lp/initial_lp < 1:
            step_size *= lr_increase_factor
        else:
            step_size *= lr_decrease_factor

        # update initial lp
        initial_lp = lp

        # update number of runs
        runs += 1

    # In optimise mode, just return RRMSE values
    if optimise:
        return RRMSE(noiseless_img, x)

    # otherwise save all images and plots
    else:
        plot_images(x,noisy_img,noiseless_img,lp_values,runs,prior,dir)


