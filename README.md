# Vehicle counting
Count vehicle using density map regression.

## Installation
```
pip3 install requirements.txt
```

## CityCam dataset
Download the dataset from the link down below. After that, extract the datasets in the project folder. Each car in the frames is annotated by a bounding box.  
[CityCam dataset](https://www.citycam-cmu.com/dataset)

**Note that CityCam contains some bad-formed xml files, in which the annotator failed to escape '&'. I also found that in one specific image frame the 'y_min' of the bounding box is out of the image border. All of invalid data I found has been listed in the `CityCam_invalid_frames.txt`.**

## Density map
<img src="imgs/og_img.png" alt="original image frame" width="352"/>
<img src="imgs/density_map.png" alt="generated density map" width="352"/>

Gernally speaking, I put a gaussian kernel at the center of every car to generate a density map for a specific image frame. As for the standard deviation (sigma) of the gaussian kernel, it is determined by the sum of the distances to the k nearest neighbors. Please read this [MCNN paper](https://www.semanticscholar.org/paper/Single-Image-Crowd-Counting-via-Multi-Column-Neural-Zhang-Zhou/427d6d9bc05b07c85fc6b2e52f12132f79a28f6c) for more details. You could find my implementation over here [k_nearest_gaussian_kernel](https://github.com/MartinMa28/vehicle_counting/blob/master/density_map/k_nearest_gaussian_kernel.py).

To generate the density maps, just type `python3 counting_datasets/CityCam_maker.py` in the project directory.

## How to train
First of all, set up your hyper parameters in hyper_param_conf.py. And then, execute the main.py script `python3 main.py`. The trained model will be stored in checkpoints/, logs will be saved in logs/ for your reference.
