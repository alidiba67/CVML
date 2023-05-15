# Computer Vision and Machine Learning: All you need to know [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Welcome to the Computer Vision and Machine Learning Research Repository! This repository aims to provide a comprehensive list of important topics and the most recent content related to computer vision and deep neural network research. Whether you are a researcher, student, or enthusiast, this repository will serve as a valuable resource for staying up-to-date with the latest advancements in the field.

## Table of Contents

- [Image Classification](#image-classification)
- [Object Detection](#object-detection)
- [Image Segmentation](#image-segmentation)
- [Action Classification](#action-classification)
- [Video Recognition](#video-recognition)
- [Deep CNN Models](#deep-cnn-models)
- [3D Vision Models](#3d-vision-models)
- [Additional Resources](#additional-resources)
- [Contributing](#contributing)
- [License](#license)

## Image Classification

In this section, you will find resources related to image classification, including datasets, models, benchmarking techniques, and recent research papers.

#### VGG Test
**Very Deep Convolutional Networks for Large-Scale Image Recognition.**
Karen Simonyan, Andrew Zisserman
- pdf: [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
- code: [torchvision : https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)

#### GoogleNet
**Going Deeper with Convolutions**
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
- pdf: [https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)
- code: [unofficial-tensorflow : https://github.com/conan7882/GoogLeNet-Inception](https://github.com/conan7882/GoogLeNet-Inception)
- code: [unofficial-caffe : https://github.com/lim0606/caffe-googlenet-bn](https://github.com/lim0606/caffe-googlenet-bn)

#### PReLU-nets
**Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification**
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- pdf: [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
- code: [unofficial-chainer : https://github.com/nutszebra/prelu_net](https://github.com/nutszebra/prelu_net)

#### ResNet
**Deep Residual Learning for Image Recognition**
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- pdf: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- code: [facebook-torch : https://github.com/facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)
- code: [torchvision : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet.py)
- code: [unofficial-keras : https://github.com/raghakot/keras-resnet](https://github.com/raghakot/keras-resnet)
- code: [unofficial-tensorflow : https://github.com/ry/tensorflow-resnet](https://github.com/ry/tensorflow-resnet)

#### PreActResNet
**Identity Mappings in Deep Residual Networks**
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- pdf: [https://arxiv.org/abs/1603.05027](https://arxiv.org/abs/1603.05027)
- code: [facebook-torch : https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua)
- code: [official : https://github.com/KaimingHe/resnet-1k-layers](https://github.com/KaimingHe/resnet-1k-layers)
- code: [unoffical-pytorch : https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py](https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py)
- code: [unoffical-mxnet : https://github.com/tornadomeet/ResNet](https://github.com/tornadomeet/ResNet)

#### Inceptionv3
**Rethinking the Inception Architecture for Computer Vision**
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
- pdf: [https://arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567)
- code: [torchvision : https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py)

#### Inceptionv4 && Inception-ResNetv2
**Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning**
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
- pdf: [https://arxiv.org/abs/1602.07261](https://arxiv.org/abs/1602.07261)
- code: [unofficial-keras : https://github.com/kentsommer/keras-inceptionV4](https://github.com/kentsommer/keras-inceptionV4)
- code: [unofficial-keras : https://github.com/titu1994/Inception-v4](https://github.com/titu1994/Inception-v4)
- code: [unofficial-keras : https://github.com/yuyang-huang/keras-inception-resnet-v2](https://github.com/yuyang-huang/keras-inception-resnet-v2)

#### RiR
**Resnet in Resnet: Generalizing Residual Architectures**
Sasha Targ, Diogo Almeida, Kevin Lyman
- pdf: [https://arxiv.org/abs/1603.08029](https://arxiv.org/abs/1603.08029)
- code: [unofficial-tensorflow : https://github.com/SunnerLi/RiR-Tensorflow](https://github.com/SunnerLi/RiR-Tensorflow)
- code: [unofficial-chainer : https://github.com/nutszebra/resnet_in_resnet](https://github.com/nutszebra/resnet_in_resnet)

#### Stochastic Depth ResNet
**Deep Networks with Stochastic Depth**
Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger
- pdf: [https://arxiv.org/abs/1603.09382](https://arxiv.org/abs/1603.09382)
- code: [unofficial-torch : https://github.com/yueatsprograms/Stochastic_Depth](https://github.com/yueatsprograms/Stochastic_Depth)
- code: [unofficial-chainer : https://github.com/yasunorikudo/chainer-ResDrop](https://github.com/yasunorikudo/chainer-ResDrop)
- code: [unofficial-keras : https://github.com/dblN/stochastic_depth_keras](https://github.com/dblN/stochastic_depth_keras)

#### WRN
**Wide Residual Networks**
Sergey Zagoruyko, Nikos Komodakis
- pdf: [https://arxiv.org/abs/1605.07146](https://arxiv.org/abs/1605.07146)
- code: [official : https://github.com/szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)
- code: [unofficial-pytorch : https://github.com/xternalz/WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch)
- code: [unofficial-keras : https://github.com/asmith26/wide_resnets_keras](https://github.com/asmith26/wide_resnets_keras)
- code: [unofficial-pytorch : https://github.com/meliketoy/wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)

#### SqueezeNet
**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size**
Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
- pdf: [https://arxiv.org/abs/1602.07360](https://arxiv.org/abs/1602.07360)
- code: [torchvision : https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py)
- code: [unofficial-caffe : https://github.com/DeepScale/SqueezeNet](https://github.com/DeepScale/SqueezeNet)
- code: [unofficial-keras : https://github.com/rcmalli/keras-squeezenet](https://github.com/rcmalli/keras-squeezenet)
- code: [unofficial-caffe : https://github.com/songhan/SqueezeNet-Residual](https://github.com/songhan/SqueezeNet-Residual)

#### GeNet
**Genetic CNN**
Lingxi Xie, Alan Yuille
- pdf: [https://arxiv.org/abs/1703.01513](https://arxiv.org/abs/1703.01513)
- code: [unofficial-tensorflow : https://github.com/aqibsaeed/Genetic-CNN](https://github.com/aqibsaeed/Genetic-CNN)

#### MetaQNN
**Designing Neural Network Architectures using Reinforcement Learning**
Bowen Baker, Otkrist Gupta, Nikhil Naik, Ramesh Raskar
- pdf: [https://arxiv.org/abs/1611.02167](https://arxiv.org/abs/1611.02167)
- code: [official : https://github.com/bowenbaker/metaqnn](https://github.com/bowenbaker/metaqnn)

##### PyramidNet
**Deep Pyramidal Residual Networks**
Dongyoon Han, Jiwhan Kim, Junmo Kim
- pdf: [https://arxiv.org/abs/1610.02915](https://arxiv.org/abs/1610.02915)
- code: [official : https://github.com/jhkim89/PyramidNet](https://github.com/jhkim89/PyramidNet)
- code: [unofficial-pytorch : https://github.com/dyhan0920/PyramidNet-PyTorch](https://github.com/dyhan0920/PyramidNet-PyTorch)

##### DenseNet
**Densely Connected Convolutional Networks**
Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
- pdf: [https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
- code: [official : https://github.com/liuzhuang13/DenseNet](https://github.com/liuzhuang13/DenseNet)
- code: [unofficial-keras : https://github.com/titu1994/DenseNet](https://github.com/titu1994/DenseNet)
- code: [unofficial-caffe : https://github.com/shicai/DenseNet-Caffe](https://github.com/shicai/DenseNet-Caffe)
- code: [unofficial-tensorflow : https://github.com/YixuanLi/densenet-tensorflow](https://github.com/YixuanLi/densenet-tensorflow)
- code: [unofficial-pytorch : https://github.com/YixuanLi/densenet-tensorflow](https://github.com/YixuanLi/densenet-tensorflow)
- code: [unofficial-pytorch : https://github.com/bamos/densenet.pytorch](https://github.com/bamos/densenet.pytorch)
- code: [unofficial-keras : https://github.com/flyyufelix/DenseNet-Keras](https://github.com/flyyufelix/DenseNet-Keras)

##### FractalNet
**FractalNet: Ultra-Deep Neural Networks without Residuals**
Gustav Larsson, Michael Maire, Gregory Shakhnarovich
- pdf: [https://arxiv.org/abs/1605.07648](https://arxiv.org/abs/1605.07648)
- code: [unofficial-caffe : https://github.com/gustavla/fractalnet](https://github.com/gustavla/fractalnet)
- code: [unofficial-keras : https://github.com/snf/keras-fractalnet](https://github.com/snf/keras-fractalnet)
- code: [unofficial-tensorflow : https://github.com/tensorpro/FractalNet](https://github.com/tensorpro/FractalNet)

##### ResNext
**Aggregated Residual Transformations for Deep Neural Networks**
Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
- pdf: [https://arxiv.org/abs/1611.05431](https://arxiv.org/abs/1611.05431)
- code: [official : https://github.com/facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnext.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnext.py)
- code: [unofficial-pytorch : https://github.com/prlz77/ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch)
- code: [unofficial-keras : https://github.com/titu1994/Keras-ResNeXt](https://github.com/titu1994/Keras-ResNeXt)
- code: [unofficial-tensorflow : https://github.com/taki0112/ResNeXt-Tensorflow](https://github.com/taki0112/ResNeXt-Tensorflow)
- code: [unofficial-tensorflow : https://github.com/wenxinxu/ResNeXt-in-tensorflow](https://github.com/wenxinxu/ResNeXt-in-tensorflow)

##### IGCV1
**Interleaved Group Convolutions for Deep Neural Networks**
Ting Zhang, Guo-Jun Qi, Bin Xiao, Jingdong Wang
- pdf: [https://arxiv.org/abs/1707.02725](https://arxiv.org/abs/1707.02725)
- code [official : https://github.com/hellozting/InterleavedGroupConvolutions](https://github.com/hellozting/InterleavedGroupConvolutions)

##### Residual Attention Network
**Residual Attention Network for Image Classification**
Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang
- pdf: [https://arxiv.org/abs/1704.06904](https://arxiv.org/abs/1704.06904)
- code: [official : https://github.com/fwang91/residual-attention-network](https://github.com/fwang91/residual-attention-network)
- code: [unofficial-pytorch : https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch](https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch)
- code: [unofficial-gluon : https://github.com/PistonY/ResidualAttentionNetwork](https://github.com/PistonY/ResidualAttentionNetwork)
- code: [unofficial-keras : https://github.com/koichiro11/residual-attention-network](https://github.com/koichiro11/residual-attention-network)

#### Xception
**Xception: Deep Learning with Depthwise Separable Convolutions**
François Chollet
- pdf: [https://arxiv.org/abs/1610.02357](https://arxiv.org/abs/1610.02357)
- code: [unofficial-pytorch : https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py](https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py)
- code: [unofficial-tensorflow : https://github.com/kwotsin/TensorFlow-Xception](https://github.com/kwotsin/TensorFlow-Xception)
- code: [unofficial-caffe : https://github.com/yihui-he/Xception-caffe](https://github.com/yihui-he/Xception-caffe)
- code: [unofficial-pytorch : https://github.com/tstandley/Xception-PyTorch](https://github.com/tstandley/Xception-PyTorch)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py)

#### MobileNet
**MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
- pdf: [https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)
- code: [unofficial-tensorflow : https://github.com/Zehaos/MobileNet](https://github.com/Zehaos/MobileNet)
- code: [unofficial-caffe : https://github.com/shicai/MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe)
- code: [unofficial-pytorch : https://github.com/marvis/pytorch-mobilenet](https://github.com/marvis/pytorch-mobilenet)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py)

#### PolyNet
**PolyNet: A Pursuit of Structural Diversity in Very Deep Networks**
Xingcheng Zhang, Zhizhong Li, Chen Change Loy, Dahua Lin
- pdf: [https://arxiv.org/abs/1611.05725](https://arxiv.org/abs/1611.05725)
- code: [official : https://github.com/open-mmlab/polynet](https://github.com/open-mmlab/polynet)

#### DPN
**Dual Path Networks**
Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng
- pdf: [https://arxiv.org/abs/1707.01629](https://arxiv.org/abs/1707.01629)
- code: [official : https://github.com/cypw/DPNs](https://github.com/cypw/DPNs)
- code: [unoffical-keras : https://github.com/titu1994/Keras-DualPathNetworks](https://github.com/titu1994/Keras-DualPathNetworks)
- code: [unofficial-pytorch : https://github.com/oyam/pytorch-DPNs](https://github.com/oyam/pytorch-DPNs)
- code: [unofficial-pytorch : https://github.com/rwightman/pytorch-dpn-pretrained](https://github.com/rwightman/pytorch-dpn-pretrained)

#### Block-QNN
**Practical Block-wise Neural Network Architecture Generation**
Zhao Zhong, Junjie Yan, Wei Wu, Jing Shao, Cheng-Lin Liu
- pdf: [https://arxiv.org/abs/1708.05552](https://arxiv.org/abs/1708.05552)

#### CRU-Net
**Sharing Residual Units Through Collective Tensor Factorization in Deep Neural Networks**
Chen Yunpeng, Jin Xiaojie, Kang Bingyi, Feng Jiashi, Yan Shuicheng
- pdf: [https://arxiv.org/abs/1703.02180](https://arxiv.org/abs/1703.02180)
- code [official : https://github.com/cypw/CRU-Net](https://github.com/cypw/CRU-Net)
- code [unofficial-mxnet : https://github.com/bruinxiong/Modified-CRUNet-and-Residual-Attention-Network.mxnet](https://github.com/bruinxiong/Modified-CRUNet-and-Residual-Attention-Network.mxnet)

## DLA
**Deep Layer Aggregation**
Fisher Yu, Dequan Wang, Evan Shelhamer, Trevor Darrell
- pdf: [https://arxiv.org/abs/1707.06484](https://arxiv.org/abs/1707.06484)
- code: [official-pytorch: https://github.com/ucbdrive/dla](https://github.com/ucbdrive/dla)

#### ShuffleNet
**ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices**
Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
- pdf: [https://arxiv.org/abs/1707.01083](https://arxiv.org/abs/1707.01083)
- code: [unofficial-tensorflow : https://github.com/MG2033/ShuffleNet](https://github.com/MG2033/ShuffleNet)
- code: [unofficial-pytorch : https://github.com/jaxony/ShuffleNet](https://github.com/jaxony/ShuffleNet)
- code: [unofficial-caffe : https://github.com/farmingyard/ShuffleNet](https://github.com/farmingyard/ShuffleNet)
- code: [unofficial-keras : https://github.com/scheckmedia/keras-shufflenet](https://github.com/scheckmedia/keras-shufflenet)

#### CondenseNet
**CondenseNet: An Efficient DenseNet using Learned Group Convolutions**
Gao Huang, Shichen Liu, Laurens van der Maaten, Kilian Q. Weinberger
- pdf: [https://arxiv.org/abs/1711.09224](https://arxiv.org/abs/1711.09224)
- code: [official : https://github.com/ShichenLiu/CondenseNet](https://github.com/ShichenLiu/CondenseNet)
- code: [unofficial-tensorflow : https://github.com/markdtw/condensenet-tensorflow](https://github.com/markdtw/condensenet-tensorflow)

#### NasNet
**Learning Transferable Architectures for Scalable Image Recognition**
Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le
- pdf: [https://arxiv.org/abs/1707.07012](https://arxiv.org/abs/1707.07012)
- code: [unofficial-keras : https://github.com/titu1994/Keras-NASNet](https://github.com/titu1994/Keras-NASNet)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/nasnet.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/nasnet.py)
- code: [unofficial-pytorch : https://github.com/wandering007/nasnet-pytorch](https://github.com/wandering007/nasnet-pytorch)
- code: [unofficial-tensorflow : https://github.com/yeephycho/nasnet-tensorflow](https://github.com/yeephycho/nasnet-tensorflow)

#### MobileNetV2
**MobileNetV2: Inverted Residuals and Linear Bottlenecks**
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
- pdf: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- code: [unofficial-keras : https://github.com/xiaochus/MobileNetV2](https://github.com/xiaochus/MobileNetV2)
- code: [unofficial-pytorch : https://github.com/Randl/MobileNetV2-pytorch](https://github.com/Randl/MobileNetV2-pytorch)
- code: [unofficial-tensorflow : https://github.com/neuleaf/MobileNetV2](https://github.com/neuleaf/MobileNetV2)

#### IGCV2
**IGCV2: Interleaved Structured Sparse Convolutional Neural Networks**
Guotian Xie, Jingdong Wang, Ting Zhang, Jianhuang Lai, Richang Hong, Guo-Jun Qi
- pdf: [https://arxiv.org/abs/1804.06202](https://arxiv.org/abs/1804.06202)

#### hier
**Hierarchical Representations for Efficient Architecture Search**
Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, Koray Kavukcuoglu
- pdf: [https://arxiv.org/abs/1711.00436](https://arxiv.org/abs/1711.00436)

#### PNasNet
**Progressive Neural Architecture Search**
Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, Kevin Murphy
- pdf: [https://arxiv.org/abs/1712.00559](https://arxiv.org/abs/1712.00559)
- code: [tensorflow-slim : https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/pnasnet.py](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/pnasnet.py)
- code: [unofficial-pytorch : https://github.com/chenxi116/PNASNet.pytorch](https://github.com/chenxi116/PNASNet.pytorch)
- code: [unofficial-tensorflow : https://github.com/chenxi116/PNASNet.TF](https://github.com/chenxi116/PNASNet.TF)

#### AmoebaNet
**Regularized Evolution for Image Classifier Architecture Search**
Esteban Real, Alok Aggarwal, Yanping Huang, Quoc V Le
- pdf: [https://arxiv.org/abs/1802.01548](https://arxiv.org/abs/1802.01548)
- code: [tensorflow-tpu : https://github.com/tensorflow/tpu/tree/master/models/official/amoeba_net](https://github.com/tensorflow/tpu/tree/master/models/official/amoeba_net)

#### SENet
**Squeeze-and-Excitation Networks**
Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
- pdf: [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)
- code: [official : https://github.com/hujie-frank/SENet](https://github.com/hujie-frank/SENet)
- code: [unofficial-pytorch : https://github.com/moskomule/senet.pytorch](https://github.com/moskomule/senet.pytorch)
- code: [unofficial-tensorflow : https://github.com/taki0112/SENet-Tensorflow](https://github.com/taki0112/SENet-Tensorflow)
- code: [unofficial-caffe : https://github.com/shicai/SENet-Caffe](https://github.com/shicai/SENet-Caffe)
- code: [unofficial-mxnet : https://github.com/bruinxiong/SENet.mxnet](https://github.com/bruinxiong/SENet.mxnet)

#### ShuffleNetV2
**ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design**
Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
- pdf: [https://arxiv.org/abs/1807.11164](https://arxiv.org/abs/1807.11164)
- code: [unofficial-pytorch : https://github.com/Randl/ShuffleNetV2-pytorch](https://github.com/Randl/ShuffleNetV2-pytorch)
- code: [unofficial-keras : https://github.com/opconty/keras-shufflenetV2](https://github.com/opconty/keras-shufflenetV2)
- code: [unofficial-pytorch : https://github.com/Bugdragon/ShuffleNet_v2_PyTorch](https://github.com/Bugdragon/ShuffleNet_v2_PyTorch)
- code: [unofficial-caff2: https://github.com/wolegechu/ShuffleNetV2.Caffe2](https://github.com/wolegechu/ShuffleNetV2.Caffe2)

#### CBAM
CBAM: Convolutional Block Attention Module
Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
- pdf: [https://arxiv.org/abs/1807.06521](https://arxiv.org/abs/1807.06521)
- code: [official-pytorch : https://github.com/Jongchan/attention-module](https://github.com/Jongchan/attention-module)
- code: [unofficial-pytorch : https://github.com/luuuyi/CBAM.PyTorch](https://github.com/luuuyi/CBAM.PyTorch)
- code: [unofficial-pytorch : https://github.com/elbuco1/CBAM](https://github.com/elbuco1/CBAM)
- code: [unofficial-keras : https://github.com/kobiso/CBAM-keras](https://github.com/kobiso/CBAM-keras)


#### IGCV3
**IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks**
Ke Sun, Mingjie Li, Dong Liu, Jingdong Wang
- pdf: [https://arxiv.org/abs/1806.00178](https://arxiv.org/abs/1806.00178)
- code: [official : https://github.com/homles11/IGCV3](https://github.com/homles11/IGCV3)
- code: [unofficial-pytorch : https://github.com/xxradon/IGCV3-pytorch](https://github.com/xxradon/IGCV3-pytorch)
- code: [unofficial-tensorflow : https://github.com/ZHANG-SHI-CHANG/IGCV3](https://github.com/ZHANG-SHI-CHANG/IGCV3)

#### BAM
**BAM: Bottleneck Attention Module**
Jongchan Park, Sanghyun Woo, Joon-Young Lee, In So Kweon
- pdf: [https://arxiv.org/abs/1807.06514](https://arxiv.org/abs/1807.06514)
- code: [official-pytorch : https://github.com/Jongchan/attention-module](https://github.com/Jongchan/attention-module)
- code: [unofficial-tensorflow : https://github.com/huyz1117/BAM](https://github.com/huyz1117/BAM)

#### MNasNet
**MnasNet: Platform-Aware Neural Architecture Search for Mobile**
Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Quoc V. Le
- pdf: [https://arxiv.org/abs/1807.11626](https://arxiv.org/abs/1807.11626)
- code: [unofficial-pytorch : https://github.com/AnjieZheng/MnasNet-PyTorch](https://github.com/AnjieZheng/MnasNet-PyTorch)
- code: [unofficial-caffe : https://github.com/LiJianfei06/MnasNet-caffe](https://github.com/LiJianfei06/MnasNet-caffe)
- code: [unofficial-MxNet : https://github.com/chinakook/Mnasnet.MXNet](https://github.com/chinakook/Mnasnet.MXNet)
- code: [unofficial-keras : https://github.com/Shathe/MNasNet-Keras-Tensorflow](https://github.com/Shathe/MNasNet-Keras-Tensorflow)

#### SKNet
**Selective Kernel Networks**
Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang
- pdf: [https://arxiv.org/abs/1903.06586](https://arxiv.org/abs/1903.06586)
- code: [official : https://github.com/implus/SKNet](https://github.com/implus/SKNet)

#### DARTS
**DARTS: Differentiable Architecture Search**
Hanxiao Liu, Karen Simonyan, Yiming Yang
- pdf: [https://arxiv.org/abs/1806.09055](https://arxiv.org/abs/1806.09055)
- code: [official : https://github.com/quark0/darts](https://github.com/quark0/darts)
- code: [unofficial-pytorch : https://github.com/khanrc/pt.darts](https://github.com/khanrc/pt.darts)
- code: [unofficial-tensorflow : https://github.com/NeroLoh/darts-tensorflow](https://github.com/NeroLoh/darts-tensorflow)

#### ProxylessNAS
**ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware**
Han Cai, Ligeng Zhu, Song Han
- pdf: [https://arxiv.org/abs/1812.00332](https://arxiv.org/abs/1812.00332)
- code: [official : https://github.com/mit-han-lab/ProxylessNAS](https://github.com/mit-han-lab/ProxylessNAS)

#### MobileNetV3
**Searching for MobileNetV3**
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
- pdf: [https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)
- code: [unofficial-pytorch : https://github.com/xiaolai-sqlai/mobilenetv3](https://github.com/xiaolai-sqlai/mobilenetv3)
- code: [unofficial-pytorch : https://github.com/kuan-wang/pytorch-mobilenet-v3](https://github.com/kuan-wang/pytorch-mobilenet-v3)
- code: [unofficial-pytorch : https://github.com/leaderj1001/MobileNetV3-Pytorch](https://github.com/leaderj1001/MobileNetV3-Pytorch)
- code: [unofficial-pytorch : https://github.com/d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)
- code: [unofficial-caffe : https://github.com/jixing0415/caffe-mobilenet-v3](https://github.com/jixing0415/caffe-mobilenet-v3)
- code: [unofficial-keras : https://github.com/xiaochus/MobileNetV3](https://github.com/xiaochus/MobileNetV3)

#### Res2Net
**Res2Net: A New Multi-scale Backbone Architecture**
Shang-Hua Gao, Ming-Ming Cheng, Kai Zhao, Xin-Yu Zhang, Ming-Hsuan Yang, Philip Torr
- pdf: [https://arxiv.org/abs/1904.01169](https://arxiv.org/abs/1904.01169)
- code: [unofficial-pytorch : https://github.com/4uiiurz1/pytorch-res2net](https://github.com/4uiiurz1/pytorch-res2net)
- code: [unofficial-keras : https://github.com/fupiao1998/res2net-keras](https://github.com/fupiao1998/res2net-keras)
- code: [official-pytorch : https://github.com/Res2Net](https://github.com/Res2Net)

#### LIP-ResNet
**LIP: Local Importance-based Pooling**
Ziteng Gao, Limin Wang, Gangshan Wu
- pdf: [https://arxiv.org/abs/1908.04156](https://arxiv.org/abs/1908.04156)
- code: [official-pytorch : https://github.com/sebgao/LIP](https://github.com/sebgao/LIP)

#### EfficientNet

**EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**
Mingxing Tan, Quoc V. Le
- pdf: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
- code: [unofficial-pytorch : https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- code: [official-tensorflow : https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)


#### FixResNeXt 
**Fixing the train-test resolution discrepancy**
Hugo Touvron, Andrea Vedaldi, Matthijs Douze, Hervé Jégou
- pdf: [https://arxiv.org/abs/1906.06423](https://arxiv.org/abs/1906.06423)
- code: [official-pytorch : https://github.com/facebookresearch/FixRes](https://github.com/facebookresearch/FixRes)


#### BiT
**Big Transfer (BiT): General Visual Representation Learning**
Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby
- pdf: [https://arxiv.org/abs/1912.11370](https://arxiv.org/abs/1912.11370)
- code: [official-tensorflow: https://github.com/google-research/big_transfer](https://github.com/google-research/big_transfer)

#### PSConv + ResNext101
**PSConv: Squeezing Feature Pyramid into One Compact Poly-Scale Convolutional Layer**
Duo Li1, Anbang Yao2B, and Qifeng Chen1B
- pdf: [https://arxiv.org/abs/2007.06191](https://arxiv.org/abs/2007.06191)
- code: [https://github.com/d-li14/PSConv](https://github.com/d-li14/PSConv)


#### NoisyStudent
**Self-training with Noisy Student improves ImageNet classification**
Qizhe Xie, Minh-Thang Luong, Eduard Hovy, Quoc V. Le
- pdf: [https://arxiv.org/abs/1911.04252](https://arxiv.org/abs/1911.04252)
- code: [official-tensorflow: https://github.com/google-research/noisystudent](https://github.com/google-research/noisystudent)
- code: [unofficial-pytorch: https://github.com/sally20921/NoisyStudent](https://github.com/sally20921/NoisyStudent)

#### RegNet
**Designing Network Design Spaces**
Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár
- pdf: [https://arxiv.org/abs/2003.13678](https://arxiv.org/abs/2003.13678)
- code: [official-pytorch: https://github.com/facebookresearch/pycls](https://github.com/facebookresearch/pycls)
- code: [unofficial-pytorch: https://github.com/d-li14/regnet.pytorch](https://github.com/d-li14/regnet.pytorch)

#### GhostNet
**GhostNet: More Features from Cheap Operations**
Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu
- pdf: [https://arxiv.org/abs/1911.11907](https://arxiv.org/abs/1911.11907)
- code: [official-pytorch: https://github.com/huawei-noah/ghostnet](https://github.com/huawei-noah/ghostnet)

#### ViT
**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
- pdf: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- code: [official-tensorflow: https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)
- code: [unofficial-pytorch: https://github.com/jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)

#### DeiT
**Training data-efficient image transformers & distillation through attention**
Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou
- pdf: [https://arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877)
- code: [official-pytorch: https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)

#### PVT
**Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions**
Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao
- pdf: [https://arxiv.org/abs/2102.12122](https://arxiv.org/abs/2102.12122)
- code: [official-pytorch: https://github.com/whai362/PVT](https://github.com/whai362/PVT)

#### T2T
**Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet**
Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis EH Tay, Jiashi Feng, Shuicheng Yan
- pdf: [https://arxiv.org/abs/2101.11986](https://arxiv.org/abs/2101.11986)
- code: [official-pytorch: https://github.com/yitu-opensource/T2T-ViT](https://github.com/yitu-opensource/T2T-ViT)

#### DeepVit
**DeepViT: Towards Deeper Vision Transformer**
Daquan Zhou, Bingyi Kang, Xiaojie Jin, Linjie Yang, Xiaochen Lian, Zihang Jiang, Qibin Hou, and Jiashi Feng.
- pdf: [https://arxiv.org/abs/2103.11886](https://arxiv.org/abs/2103.11886)
- code: [official-pytorch: https://github.com/zhoudaquan/dvit_repo](https://github.com/zhoudaquan/dvit_repo)

#### ViL
**Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding**
Pengchuan Zhang, Xiyang Dai, Jianwei Yang, Bin Xiao, Lu Yuan, Lei Zhang, Jianfeng Gao
- pdf: [https://arxiv.org/abs/2103.15358](https://arxiv.org/abs/2103.15358)
- code: [official-pytorch: https://github.com/microsoft/vision-longformer](https://github.com/microsoft/vision-longformer)

#### TNT
**Transformer in Transformer**
Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, Yunhe Wang
- pdf: [https://arxiv.org/abs/2103.00112](https://arxiv.org/abs/2103.00112)
- code: [https://github.com/huawei-noah/CV-Backbones](https://github.com/huawei-noah/CV-Backbones)

#### CvT
**CvT: Introducing Convolutions to Vision Transformers**
Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang
- pdf: [https://arxiv.org/abs/2103.15808](https://arxiv.org/abs/2103.15808)
- code: [https://github.com/microsoft/CvT](https://github.com/microsoft/CvT)

#### CViT
**CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification**
Chun-Fu (Richard) Chen, Quanfu Fan, Rameswar Panda
- pdf: [https://arxiv.org/abs/2103.14899](https://arxiv.org/abs/2103.14899)
- code: [https://github.com/IBM/CrossViT](https://github.com/IBM/CrossViT)

#### Focal-T
**Focal Attention for Long-Range Interactions in Vision Transformers**
Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, Jianfeng Gao
- pdf: [https://arxiv.org/abs/2107.00641](https://arxiv.org/abs/2107.00641)
- code: [ https://github.com/microsoft/Focal-Transformer](https://github.com/microsoft/Focal-Transformer)

#### Twins
**Twins: Revisiting the Design of Spatial Attention in Vision Transformers**
- pdf: [https://arxiv.org/abs/2104.13840](https://arxiv.org/abs/2104.13840)
- code: [https://git.io/Twins]( https://git.io/Twins)

#### PVTv2
**Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao**
- pdf: [https://arxiv.org/abs/2106.13797](https://arxiv.org/abs/2106.13797)
- code: [official-pytorch: https://github.com/whai362/PVT](https://github.com/whai362/PVT)

## Object Detection

This section focuses on object detection, covering topics such as algorithms, frameworks, datasets, and state-of-the-art models developed for accurate and efficient object detection.

## Image Segmentation

Explore the field of image segmentation, including popular segmentation algorithms, datasets, and cutting-edge approaches for segmenting objects within images.

## Action Classification

Discover resources related to action classification, which involves recognizing and classifying human actions in videos. This section includes datasets, models, and techniques specific to action classification tasks.

## Video Recognition

In this section, you will find information on video recognition, which involves understanding and analyzing videos to recognize and interpret various visual elements. Explore datasets, models, and techniques for video understanding tasks.

## Deep CNN Models

This section is dedicated to deep convolutional neural network (CNN) models, which have revolutionized computer vision research. Discover well-known CNN architectures, pre-trained models, and research papers showcasing the advancements in deep learning for computer vision.

## 3D Vision Models

Explore the fascinating world of 3D vision models. This section covers topics such as 3D object recognition, depth estimation, point cloud processing, and related research papers.

## Additional Resources

For an extensive list of computer vision resources, you can also check out the [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision) repository by [Jianbo Shi](https://github.com/jbhuang0604). It contains a curated collection of various computer vision topics, including datasets, software, tutorials, and research papers.

## Contributing

We welcome contributions from the community to make this repository even more comprehensive and up-to-date. If you have any suggestions, please feel free to open an issue or submit a pull request.

## License

This repository is licensed under the [MIT License](LICENSE).
