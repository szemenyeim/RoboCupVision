# RoboCupVision
Deep Learning for Semantic Segmentation in RoboCup

This set of scripts allows you to train deep convolutional networks for semantic segmentation in Pytorch as described in our paper. This code is finetuned to work well with the networks and the problem (robot soccer) we discussed in the paper, still it should work reasonably fine for other problems as well.

## 1.Requirements

This code is written in python 2.7. The following python packages are required:

- Pytorch
- Torchvision
- Progressbar 2
- PIL Image library
- Numpy
- Visdom

You can install all of these by entering pip install &lt;package name&gt; into the console.

In order to run the code, you also need our datasets, which you can download here. Extract it to the data subfolder. (NOTE: If after extraction you have &quot;data/data/many folders&quot;, then you need to move the whole thing one folder up to get rid of one of the &quot;data&quot;-s.)

## 2.Dataset and Pre-trained models

You can download the datasets used for training and finetuning from [our website](http://deeplearning.iit.bme.hu/ROBOSeg/).

If you do not want to train your own models, just use our pretrained models, which you can download from [here](http://deeplearning.iit.bme.hu/ROBOSeg/checkpoints.zip). Extract them into the &quot;pth&quot; folder.

## 3.Training a segmentation network

In order to train the network, you&#39;ll have to pretrain the first half on a classification set first.

        python classTrainer.py

The pretrain the whole segmentation network on synthetic data.

        python trainer.py

Then finetune the network on the real dataset, and after that you can optionally prune an retrain the network.

        python trainer.py –-finetune

        python trainer.py –-finetune –-prune

The finetuned flag also determines if the synthetic or the real test sets are evaluated.

Alternatively, you can use iterative pruning on a finetuned model:

        python pruner.py

There are several further options you can use for training:

- --noScale        This option will run the network on 640x480 resolution instead of 160x120
- --v2        This option will train the network on the PB-FCNv2 architecture, which uses separated convolutions
- --topCam, --bottomCam        Use top or bottom camera images only. Providing both or neither flags will use both cameras.
- --noBall, --noRobot, --noGoal, --noLine        Treat balls, robot, goals or lines as background. You can use any combinations of these flags, as long as there is at least one foreground class.

Finally, you can generate the labeled images using the tester script. You can use the flags to select different neural network versions. Examples:

        python tester.py

        python tester.py –-finetuned

        python tester.py –-finetuned –-pruned

        python tester.py –-finetuned –-pruned --v2 --topCam

        python tester.py –-finetuned –-pruned --v2 --bottomCam --noGoal --noRobot

There are several further options you can use for training:

- --pruned2        Use pruned model produced using pruner.py (iterative pruning)
- --dump        Export the weights for RoboDNN. The weights will get exported to the weights folder
- --useCuda        The tester uses the CPU by default even if CUDA is available (to measure average exectuion time). This option forces the use of the GPU if available.

## 4.Training the label propagation network

The training of the label propagation network is quite similar to the segmentation networks, except that the classification pretraining step is omitted.

        python labelPropTrain.py

        python labelPropTrain.py -–finetune

        python labelPropTrain.py –-finetune –-prune

You can also generate the labeled images using the following script:

        python validLabelProp.py –-finetuned –-pruned

You can also compare the results with farneback optical flow using:

        python validLabelProp.py –-finetuned –-optFlow

Note, that the &#39;finetuned&#39; flag is needed to set the database to the real one.
