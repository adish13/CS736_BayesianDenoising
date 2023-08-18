import scipy.io
from ImageDenoising import ImageDenoise,RRMSE
from optimise import optimise
data = scipy.io.loadmat('../data/assignmentImageDenoisingPhantom.mat')

noiseless_img = data['imageNoiseless']
noisy_img =data['imageNoisy']

print(RRMSE(noiseless_img,noisy_img))

# Quadratic Prior
# optimal alpha = 0.102
print("QUADRATIC PRIOR ===============")
quadratic = ImageDenoise(noisy_img, noiseless_img, alpha=0.102, gamma=1, optimise=False, prior='quadratic',dir='../results/q1')
print("RRMSE at a* "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.102, gamma=1, optimise=True, prior='quadratic')))
print("RRMSE at 0.8a* "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.8*0.102, gamma=1, optimise=True, prior='quadratic')))
print("RRMSE at 1.2a* "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=1.2*0.102, gamma=1, optimise=True, prior='quadratic')))

# # Huber Prior
# # optimise(noisy_img,noiseless_img,alpha_iterations=100,gamma_iterations=1000,prior='huber')
# # optimal alpha = 0.887, gamma = 0.009
print("HUBER PRIOR ===============")

huber = ImageDenoise(noisy_img, noiseless_img, alpha=0.887, gamma= 0.009, optimise=False, prior='huber',dir='../results/q1')
print("RRMSE(a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.887, gamma=0.009, optimise=True, prior='huber')))
print("RRMSE(1.2a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=1.2*0.887, gamma=0.009, optimise=True, prior='huber')))
print("RRMSE(0.8a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.8*0.887, gamma=0.009, optimise=True, prior='huber')))
print("RRMSE(a*,1.2b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.887, gamma=1.2*0.009, optimise=True, prior='huber')))
print("RRMSE(a*,0.8b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.887, gamma=0.8*0.009, optimise=True, prior='huber')))

# # Adaptive Prior
# optimise(noisy_img,noiseless_img,alpha_iterations=100,gamma_iterations=100,prior='adaptive')
# # optimal alpha = 0.9, gamma = 0.008
print("ADAPTIVE PRIOR ===============")

adaptive = ImageDenoise(noisy_img, noiseless_img, alpha=0.9, gamma= 0.008, optimise=False, prior='adaptive',dir='../results/q1')
print("RRMSE(a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.9, gamma=0.008, optimise=True, prior='adaptive')))
print("RRMSE(1.2a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=1.2*0.9, gamma=0.008, optimise=True, prior='adaptive')))
print("RRMSE(0.8a*,b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.8*0.9, gamma=0.008, optimise=True, prior='adaptive')))
print("RRMSE(a*,1.2b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.9, gamma=1.2*0.008, optimise=True, prior='adaptive')))
print("RRMSE(a*,0.8b*) "+ str(ImageDenoise(noisy_img, noiseless_img, alpha=0.9, gamma=0.8*0.008, optimise=True, prior='adaptive')))


