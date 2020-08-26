# maize-yield-tsc

This is a project that aims the development of a model for yield and Tar Spot Complex (TSC) prediction using multispectral images from maize crops. We tested some neural network topologies for the development of a regression model for TSC and Yield prediction.

## Original work

This work is based on the [paper](https://www.frontiersin.org/articles/10.3389/fpls.2019.00552/full). The authors showed that TSC and Yield has a high covariance between different bands and different graphical indicators such as normalized difference vegetation index (NVDI) and others.

## Jupyter notebooks

The main work was done using jupyter notebooks. For hyperparameters search, a script was generated from the notebook to train a wide range of topologies and hyperparameters. See models.py to see some models tested.

## Data

Our data consist in two main groups of images: multispectral images (4 bands) and thermal images.

![data](https://www.frontiersin.org/files/Articles/432168/fpls-10-00552-HTML/image_m/fpls-10-00552-g002.jpg)

For additional visualization examples, see vis folder. 

![instance](https://user-images.githubusercontent.com/34692520/91305568-9a13ae80-e781-11ea-9b6c-d5037db5ae58.png)
### Data augmentation

We tested different data augmentation techniques which helped the model for a better generalization using the fours bands of the multispectral images. 

See util.py to see the data augmentation techniques applied.

## Results

We are currently working on the paper of this project. We will link the paper here in the future.

![disp](https://user-images.githubusercontent.com/34692520/91305709-cf200100-e781-11ea-92ab-c88bd55f5334.png)
