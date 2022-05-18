# RoboCupVision
Deep Learning for Semantic Segmentation in RoboCup

This set of scripts allows you to train deep convolutional networks for semantic segmentation in Pytorch as described in our paper. This code is finetuned to work well with the networks and the problem (robot soccer) we discussed in the paper, still it should work reasonably fine for other problems as well.

## 1.Requirements

This code is written in python 3.6. The following python packages are required:

- Pytorch
- Torchvision
- Progressbar 2
- PIL Image library
- Numpy
- Visdom

You can install all of these by entering pip install &lt;package name&gt; into the console.

In order to run the code, you also need our datasets, which you can download here. Extract it to the data subfolder. (NOTE: If after extraction you have &quot;data/data/many folders&quot;, then you need to move the whole thing one folder up to get rid of one of the &quot;data&quot;-s.)

## 2.Dataset and Pre-trained models

You can download the datasets used for training and finetuning from our website

- [Train set](https://deeplearning.iit.bme.hu/Public/ROBOSeg/train.zip)
- [Validation set](https://deeplearning.iit.bme.hu/Public/ROBOSeg/val.zip)
- [Fintune set](https://deeplearning.iit.bme.hu/Public/ROBOSeg/FinetuneHorizon.zip)

If you do not want to train your own models, just use our pretrained models, which you can download from [here](https://deeplearning.iit.bme.hu/Public/ROBOSeg/checkpoints.zip). Extract them into the &quot;checkpoints&quot; folder.

## 3.Training a segmentation network

The pretrain the whole segmentation network on synthetic data.

        python train.py

Then finetune the network on the real dataset, (you can use synthetic transfer learning optionally).

        python train.py –-finetune

        python train.py –-finetune --transfer

The finetune flag also determines if the synthetic or the real test sets are evaluated.

There are several further options you can use for training:

- --noScale        This option will run the network on 640x480 resolution instead of 160x120
- --UNet        This option will train the standard U-Net architecture, instead of the ROBO-UNet proposed in our paper.
- --topCam, --bottomCam        Use top or bottom camera images only. Providing both or neither flags will use both cameras.
- --noBall, --noRobot, --noGoal, --noLine        Treat balls, robot, goals or lines as background. You can use any combinations of these flags, as long as there is at least one foreground class.

Finally, you can generate the labeled images using the tester script. You can use the flags to select different neural network versions. Examples:

        python test.py

        python test.py –-finetune

        python tester.py –-finetune –-transfer

        python tester.py –-finetune --v2 --topCam

        python tester.py –-finetune --v2 --bottomCam --noGoal --noRobot

There are several further options you can use for training:

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
