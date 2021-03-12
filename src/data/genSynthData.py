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
    samplesCartesian = torch.tensor([[np.cos(thetas[i]), np.sin(thetas[i]), i] for i in range(Nclasses)]).T # Size: (Nsamples, 2, Nclasses)

    if plotCartesian:    
        # Plotting samples 
        plt.figure(figsize = (5,5))
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        for i in range(Nclasses):
            plt.plot(samplesCartesian[:,0,i], samplesCartesian[:,1,i], '.')
    
    return samplesCartesian


# A function creating synthetic data on S1
# Input: @Nsamples: The number of samples in each class, integer 
#        @NClasses: The number of classes, integer 
#        @plotCartesian: Display a plot of S1 with the points, boolean 
# Output: @samplesCartesian: Cartesian coordinates in R2 of the samples on S1. 
#                            dimension is (Nsamples, 2, Nclasses)
def genSynthDataS1_new(Nsamples = 100, Nclasses = 3, plotCartesian = True):  
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
    samplesCartesian = np.array(np.transpose([[np.cos(thetas[i]), np.sin(thetas[i]), np.ones(len(thetas[0]))*i] for i in range(Nclasses)])) # Size: (Nsamples, 3, Nclasses)
    samplesCartesian = np.transpose(np.hstack(samplesCartesian))

    return samplesCartesian