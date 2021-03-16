def genNoisySynthDataS1(Nsamples = 100, Nclasses = 3, plotCartesian = True, newDim= 20):
    # Getting synhetic S1 data 
    coords, target = genSynthDataS1(Nsamples, Nclasses, plotCartesian)

    # Multiplicative and additive random noise 
    noisyCoords = coords@np.random.normal(size = (np.shape(coords)[1],newDim)) + np.random.normal(size = (np.shape(coords)[0],newDim))

    return (noisyCoords, target)


# A function creating synthetic data on S1
# Input: @Nsamples: The number of samples in each class, integer 
#        @NClasses: The number of classes, integer 
#        @plotCartesian: Display a plot of S1 with the points, boolean 
# Output: @samplesCartesian: Cartesian coordinates in R2 of the samples on S1. 
#                            dimension is (Nsamples, 2, Nclasses)
def genSynthDataS1(Nsamples = 100, Nclasses = 3, plotCartesian = True):  
    import numpy as np  
    import matplotlib.pyplot as plt
    import seaborn as snb
    import torch
    import random

    # Seeting seed for reproducability
    np.random.seed(60220)

    #Drawing means between -pi and pi
    means = np.random.uniform(low = -np.pi, high = np.pi, size = Nclasses)
    scales = np.random.uniform(low = 0.1, high = 1, size = Nclasses)

    # Drawing angles from normal distribution 
    thetas = [np.random.normal(loc = means[i], scale = scales[i], size = Nsamples) for i in range(Nclasses)]

    # Changing to polar coordinates 
    samplesCartesian = [np.transpose(np.array([np.cos(thetas[i]), np.sin(thetas[i]),  np.ones(len(thetas[i]))*i])) for i in range(Nclasses)] # Size: samples, 2, Nclasses)
    samplesCartesian = np.concatenate(samplesCartesian)

    coords = samplesCartesian[:,0:2]
    target = samplesCartesian[:,2]

    if plotCartesian:    
        # Plotting samples 
        plt.figure(figsize = (5,5))
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.scatter(x = samplesCartesian[:,0], y = samplesCartesian[:,1], c = samplesCartesian[:,2], cmap = "tab10")
    
    return (coords, target) 


