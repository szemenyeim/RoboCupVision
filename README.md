# RoboCupVision
Deep Learning for Semantic Segmentation in RoboCup

This set of scripts allows you to train deep convolutional networks for semantic segmentation in Pytorch as described in our paper. This code is finetuned to work well with the networks and the problem (robot soccer) we discussed in the paper, still it should work reasonably fine for other problems as well.

1.Requirements

This code is written in python 2.7. The following python packages are required:

- Pytorch
- Torchvision
- Progressbar 2
- PIL Image library
- Numpy
- Visdom

You can install all of these by entering pip install &lt;package name&gt; into the console.

In order to run the code, you also need our datasets, which you can download here. Extract it to the data subfolder. (NOTE: If after extraction you have &quot;data/data/many folders&quot;, then you need to move the whole thing one folder up to get rid of one of the &quot;data&quot;-s.)

2.Pre-trained models

If you do not want to train your own models, just use our pretrained ones, you can download them here. Extract them into the &quot;pth&quot; folder.

3.Training a segmentation network

In order to train the network, you&#39;ll have to pretrain the first half on a classification set first.

        python classTrainer.py

The pretrain the whole segmentation network on synthetic data.

        python trainer.py

Then finetune the network on the real dataset, and after that you can optionally prune an retrain the network.

        python trainer.py –finetune

        python trainer.py –finetune –prune

Finally, you can generate the labeled images using the tester script. You can use the flags to select different neural network versions.

        python tester.py

python tester.py –finetuned

python tester.py –finetuned –pruned

The finetuned flag also determines if the synthetic or the real test sets are evaluated.

There are several further options you can use:

- --noScale        This option will run the network on 640x480 resolution instead of 160x120
- --deep                This option will run a ResNet152+DUC network instead of PB-FCN
- --FCN                This option will run a simple FCN network instead of the PB-FCN

4.Training the label propagation network

The training of the label propagation network is quite similar to the segmentation networks, except that the classification pretraining step is omitted.

python labelPropTrain.py

python labelPropTrain.py –finetune

python labelPropTrain.py –finetune –prune

You can also generate the labeled images using the following script:

python validLabelProp.py –finetuned –pruned

You can also compare the results with farneback optical flow using:

python validLabelProp.py –finetuned –optFlow

Note, that the &#39;finetuned&#39; flag is needed to set the database to the real one.
