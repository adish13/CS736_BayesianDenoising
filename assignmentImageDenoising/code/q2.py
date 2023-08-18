import scipy.io
from ImageDenoising import RRMSE, ImageDenoise
from optimise import optimise

data = scipy.io.loadmat('../data/brainMRIslice.mat')

noisy_img = data['brainMRIsliceNoisy']
noiseless_img = data['brainMRIsliceOrig']


print(RRMSE(noiseless_img,noisy_img))

# Quadratic Prior
# # optimal alpha = 0.133
print("QUADRATIC PRIOR ===============")

quadratic = ImageDenoise(noisy_img, noiseless_img, alpha=0.133, gamma=1, optimise=False, prior='quadratic',dir='../results/q2')
print("RRMSE at a* "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.133, gamma=1, optimise=True, prior='quadratic')))
print("RRMSE at 0.8a* "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.8*0.133, gamma=1, optimise=True, prior='quadratic')))
print("RRMSE at 1.2a* "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=1.2*0.133, gamma=1, optimise=True, prior='quadratic')))

# # Huber Prior
# # optimise(noisy_img,noiseless_img,alpha_iterations=100,gamma_iterations=1000,prior='huber')

print("HUBER PRIOR ===============")

huber = ImageDenoise(noisy_img, noiseless_img, alpha=0.42, gamma= 0.062, optimise=False, prior='huber',dir='../results/q2')
print("RRMSE(a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.42, gamma=0.062, optimise=True, prior='huber')))
print("RRMSE(1.2a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=1.2*0.42, gamma=0.062, optimise=True, prior='huber')))
print("RRMSE(0.8a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.8*0.42, gamma=0.062, optimise=True, prior='huber')))
print("RRMSE(a*,1.2b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.42, gamma=1.2*0.062, optimise=True, prior='huber')))
print("RRMSE(a*,0.8b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.42, gamma=0.8*0.062, optimise=True, prior='huber')))

# Adaptive Prior
# optimise(noisy_img,noiseless_img,alpha_iterations=100,gamma_iterations=1000,prior='adaptive')

print("ADAPTIVE PRIOR ===============")

daptive = ImageDenoise(noisy_img, noiseless_img, alpha=0.56, gamma= 0.05, optimise=False, prior='adaptive',dir='../results/q2')
print("RRMSE(a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.56, gamma=0.05, optimise=True, prior='adaptive')))
print("RRMSE(1.2a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=1.2*0.56, gamma=0.05, optimise=True, prior='adaptive')))
print("RRMSE(0.8a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.8*0.56, gamma=0.05, optimise=True, prior='adaptive')))
print("RRMSE(a*,1.2b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.56, gamma=1.2*0.05, optimise=True, prior='adaptive')))
print("RRMSE(a*,0.8b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.56, gamma=0.8*0.05, optimise=True, prior='adaptive')))

