from ImageDenoising import ImageDenoise

def optimise(noisy_img,noiseless_img,alpha_iterations = 10,gamma_iterations=10,prior='quadratic'):
    min_rrmse = 1000
    if prior == 'quadratic':
        gamma_iterations = 2
    else:
        gamma_iterations = gamma_iterations

    for i in range(89,alpha_iterations):
        for j in range(1,gamma_iterations):
            a = (1/alpha_iterations)*i
            g = (1/gamma_iterations)*j
            output = ImageDenoise(noisy_img, noiseless_img, alpha=a, gamma=g, optimise=True, prior=prior)
            print("RRMSE for alpha = {} and gamma = {} = {}".format(a,g,output))
            if (output < min_rrmse):
                min_rrmse = output
                optimum_alpha = a 
                optimum_gamma = g

    print("Min RRMSE for alpha = {} and gamma = {} = {}".format(optimum_alpha,optimum_gamma,min_rrmse))

