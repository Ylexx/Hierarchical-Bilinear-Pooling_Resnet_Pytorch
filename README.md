# Hierarchical-Bilinear-Pooling_Resnet_Pytorch

[![](https://img.shields.io/badge/HBP-Model-green.svg)](https://github.com/Ylexx/Hierarchical-Bilinear-Pooling_Resnet_pytorch)

 A reimplementation of Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition in Resnet.
 
 



## Requirements
- python 2.7
- pytorch 0.4.1

## Train

Step 1. 
- Download the resnet pre-training parameters.

- Download the CUB-200-2011 dataset.
[CUB-download](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

Step 2. 
- Set the path to the dataset and resnet parameters in the code.

Step 3. Train the fc layer only.
- python train_firststep.py


    	


Step 4. Fine-tune all layers. It gets an accuracy of around 86% on CUB-200-2011 when using resnet-50.
- python train_finetune.py



##### If you are interested in this code, you can continue to adjust the model structure, and try more resnet models, you should get better results.


##### Official Caffe implementation of Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition is [HERE](https://github.com/ChaojianYu/Hierarchical-Bilinear-Pooling)
