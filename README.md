# UnitVisSeg: Toolkit for Visualizing and Segmenting Units in Deep CNNs.

## Introduction
This repository contains the codes and results for unit visualization and segmentation. Some of the codes have been used for the [ICLR'15 paper](https://arxiv.org/pdf/1412.6856.pdf) Object Detectors Emerge in Deep Scene CNNs. You can use this toolkit with the naive [caffe](https://github.com/BVLC/caffe), with matcaffe and pycaffe compiled. This toolkit includes the following functions:

* ```generate_unitsegments.m```: Code to generate the visualization for all the units at one layer.
* ```unit_annotation```: the results of unit annotation for Places-AlexNet and ImageNet-AlexNet.
* ```unit_segmentation```: the code to generate image segmentation for a single image using the synthetic receptive field.
* ```minimal_image```: possion removal of the image content, to generate the minimal images.


## Download
* Clone the code from github
```
    git clone https://github.com/metalbubble/unitvisseg.git
    cd unitvisseg
```
* Download the image samples and the pretrained models
```
    sh download_images.sh
    sh download_pretrain.sh
``` 

## Run
Properly set up the caffe path in the shell scripts, matlab scripts, and python scripts. We show the procedure to probe the unit semantics for alexnet trained on Places205 as follows:

* Generate the visualization for all the units in one layer of a given CNN. It gets the top ranked images for each unit then segment them using the approximate receptive field of the units. The results are output to result_segments. After it finishes, you could click ```result_segments/places205_alexnet.html``` to see the visualization of unit segmentation.
```
    matlab -nodisplay -r generate_unitsegments.m
```
Samples of unit segmentation are shown below. Each row indicates the images segmented by the receptive field of one unit.
![segmentation](http://places.csail.mit.edu/unit_annotation/images/unitsegmentation_new.png)

* Generate the minimal image. 
```
    cd minimal_image
    matlab
    demo_bases
```
Sample of the bases and images generated from combining the bases:
![bases](http://places.csail.mit.edu/unit_annotation/images/minimal_image.png)

* Generate image segmentation using the synthetic receptive field of one unit.
```
    cd unit_segmentation
    matlab
    demo_unitsegments
```
It generates the image segmentation like this:
![unitsegmentation](http://places.csail.mit.edu/unit_annotation/images/sample_segment.png)

## References
If you find the toolkit useful, please cite our ICLR'15 paper:
```
    Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba
    Object Detectors Emerge in Deep Scene CNNs.
    International Conference on Learning Representations (ICLR), 2015.
```

