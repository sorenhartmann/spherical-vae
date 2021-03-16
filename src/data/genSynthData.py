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


def genSynthDataS2(Nclasses = 3, Nsamples = 200, plotCartesian = True):
    #Generating Synthetic data on S2
    import numpy as np  
    import matplotlib.pyplot as plt
    import seaborn as snb
    import torch
    import random
    import plotly.express as px
    from plotly import graph_objects as go

    # Seeting seed for reproducability
    #np.random.seed(60220)

    #Drawing means and sd for theta (inclination) between 0 and pi 
    means_theta = np.random.uniform(low = 0, high = np.pi, size = Nclasses)
    scales_theta = np.random.uniform(low = 0.1, high = 1, size = Nclasses)

    #Drawing means and sd for phi (azimuth) between -pi and pi
    means_phi = np.random.uniform(low = 0, high = 2*np.pi, size = Nclasses)
    scales_phi = np.random.uniform(low = 0.1, high = 1, size = Nclasses)

    # Drawing angles from normal distribution 
    thetas = [np.random.normal(loc = means_theta[i], scale = scales_theta[i], size = Nsamples) for i in range(Nclasses)]
    phis = [np.random.normal(loc = means_phi[i], scale = scales_phi[i], size = Nsamples) for i in range(Nclasses)]

    xcoor = lambda theta, phi: np.sin(theta)*np.cos(phi)
    ycoor = lambda theta, phi: np.sin(theta)*np.sin(phi)
    zcoor = lambda theta: np.cos(theta)

    # Changing to polar coordinates 
    samplesCartesian = [np.transpose(np.array([xcoor(thetas[i], phis[i]), ycoor(thetas[i], phis[i]), zcoor(thetas[i]), np.ones(len(thetas[i]))*i])) for i in range(Nclasses)] # Size: samples, [x,y,z,t], Nclasses)
    samplesCartesian = np.concatenate(samplesCartesian)

    if(plotCartesian):
        fig = go.Figure()
        fig.add_trace((go.Scatter3d(
            x=samplesCartesian[:,0], 
            y= samplesCartesian[:,1], 
            z= samplesCartesian[:,2], #np.zeros(np.shape(coords)[0]),
            mode="markers",
            marker = dict(color = samplesCartesian[:,3], 
                        size = 4)
        )))

        fig.update_layout(scene_aspectmode='manual',
                        scene_aspectratio=dict(x=1, y=1, z=1),
                        scene_xaxis=dict(range = (-1, 1)),
                        scene_yaxis=dict(range = (-1, 1)),
                        scene_zaxis=dict(range = (-1, 1)))
        fig.show()

    coords = samplesCartesian[:,0:3]
    target = samplesCartesian[:,3]

    return coords, target


def genNoisySynthDataS2(Nsamples = 200, Nclasses = 3, plotCartesian = True, newDim= 50):
    import numpy as np
    # Getting synhetic S1 data 
    coords, target = genSynthDataS2(Nsamples = Nsamples, Nclasses = Nclasses, plotCartesian = plotCartesian)

    # Multiplicative and additive random noise 
    noisyCoords = coords@np.random.normal(size = (np.shape(coords)[1],newDim)) + np.random.normal(size = (np.shape(coords)[0],newDim))

    return (noisyCoords, target)

 