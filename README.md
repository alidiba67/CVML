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
- [CVML Courses](#CVML-courses)
- [Additional Resources](#additional-resources)
- [Contributing](#contributing)
- [License](#license)

## Image Classification
⭐⭐⭐
In this section, you will find resources related to image classification, including datasets, models, benchmarking techniques, and recent research papers.

#### VGG
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

#### DLA
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
⭐⭐⭐
This section focuses on object detection, covering topics such as algorithms, frameworks, datasets, and state-of-the-art models developed for accurate and efficient object detection.

**Imbalance Problems in Object Detection: A Review**

- intro: under review at TPAMI
- arXiv: <https://arxiv.org/abs/1909.00169>

**Recent Advances in Deep Learning for Object Detection**

- intro: From 2013 (OverFeat) to 2019 (DetNAS)
- arXiv: <https://arxiv.org/abs/1908.03673>

**A Survey of Deep Learning-based Object Detection**

- intro：From Fast R-CNN to NAS-FPN

- arXiv：<https://arxiv.org/abs/1907.09408>

**Object Detection in 20 Years: A Survey**

- intro：This work has been submitted to the IEEE TPAMI for possible publication
- arXiv：<https://arxiv.org/abs/1905.05055>

**《Recent Advances in Object Detection in the Age of Deep Convolutional Neural Networks》**

- intro: awesome


- arXiv: https://arxiv.org/abs/1809.03193

**《Deep Learning for Generic Object Detection: A Survey》**

- intro: Submitted to IJCV 2018
- arXiv: https://arxiv.org/abs/1809.02165

#### R-CNN

**Rich feature hierarchies for accurate object detection and semantic segmentation**

- intro: R-CNN
- arxiv: <http://arxiv.org/abs/1311.2524>
- supp: <http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf>
- slides: <http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf>
- slides: <http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf>
- github: <https://github.com/rbgirshick/rcnn>
- notes: <http://zhangliliang.com/2014/07/23/paper-note-rcnn/>
- caffe-pr("Make R-CNN the Caffe detection example"): <https://github.com/BVLC/caffe/pull/482>

#### Fast R-CNN

**Fast R-CNN**

- arxiv: <http://arxiv.org/abs/1504.08083>
- slides: <http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf>
- github: <https://github.com/rbgirshick/fast-rcnn>
- github(COCO-branch): <https://github.com/rbgirshick/fast-rcnn/tree/coco>
- webcam demo: <https://github.com/rbgirshick/fast-rcnn/pull/29>
- notes: <http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/>
- notes: <http://blog.csdn.net/linj_m/article/details/48930179>
- github("Fast R-CNN in MXNet"): <https://github.com/precedenceguo/mx-rcnn>
- github: <https://github.com/mahyarnajibi/fast-rcnn-torch>
- github: <https://github.com/apple2373/chainer-simple-fast-rnn>
- github: <https://github.com/zplizzi/tensorflow-fast-rcnn>

**A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection**

- intro: CVPR 2017
- arxiv: <https://arxiv.org/abs/1704.03414>
- paper: <http://abhinavsh.info/papers/pdfs/adversarial_object_detection.pdf>
- github(Caffe): <https://github.com/xiaolonw/adversarial-frcnn>

#### Faster R-CNN

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**

- intro: NIPS 2015
- arxiv: <http://arxiv.org/abs/1506.01497>
- gitxiv: <http://www.gitxiv.com/posts/8pfpcvefDYn2gSgXk/faster-r-cnn-towards-real-time-object-detection-with-region>
- slides: <http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf>
- github(official, Matlab): <https://github.com/ShaoqingRen/faster_rcnn>
- github(Caffe): <https://github.com/rbgirshick/py-faster-rcnn>
- github(MXNet): <https://github.com/msracver/Deformable-ConvNets/tree/master/faster_rcnn>
- github(PyTorch--recommend): <https://github.com//jwyang/faster-rcnn.pytorch>
- github: <https://github.com/mitmul/chainer-faster-rcnn>
- github(Torch):: <https://github.com/andreaskoepf/faster-rcnn.torch>
- github(Torch):: <https://github.com/ruotianluo/Faster-RCNN-Densecap-torch>
- github(TensorFlow): <https://github.com/smallcorgi/Faster-RCNN_TF>
- github(TensorFlow): <https://github.com/CharlesShang/TFFRCNN>
- github(C++ demo): <https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus>
- github(Keras): <https://github.com/yhenon/keras-frcnn>
- github: <https://github.com/Eniac-Xie/faster-rcnn-resnet>
- github(C++): <https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev>

**R-CNN minus R**

- intro: BMVC 2015
- arxiv: <http://arxiv.org/abs/1506.06981>

**Faster R-CNN in MXNet with distributed implementation and data parallelization**

- github: <https://github.com/dmlc/mxnet/tree/master/example/rcnn>

**Contextual Priming and Feedback for Faster R-CNN**

- intro: ECCV 2016. Carnegie Mellon University
- paper: <http://abhinavsh.info/context_priming_feedback.pdf>
- poster: <http://www.eccv2016.org/files/posters/P-1A-20.pdf>

**An Implementation of Faster RCNN with Study for Region Sampling**

- intro: Technical Report, 3 pages. CMU
- arxiv: <https://arxiv.org/abs/1702.02138>
- github: <https://github.com/endernewton/tf-faster-rcnn>
- github: https://github.com/ruotianluo/pytorch-faster-rcnn

**Interpretable R-CNN**

- intro: North Carolina State University & Alibaba
- keywords: AND-OR Graph (AOG)
- arxiv: <https://arxiv.org/abs/1711.05226>

**Domain Adaptive Faster R-CNN for Object Detection in the Wild**

- intro: CVPR 2018. ETH Zurich & ESAT/PSI
- arxiv: <https://arxiv.org/abs/1803.03243>

#### Mask R-CNN

- arxiv: <http://arxiv.org/abs/1703.06870>
- github(Keras): https://github.com/matterport/Mask_RCNN
- github(Caffe2): https://github.com/facebookresearch/Detectron
- github(Pytorch): <https://github.com/wannabeOG/Mask-RCNN>
- github(MXNet): https://github.com/TuSimple/mx-maskrcnn
- github(Chainer): https://github.com/DeNA/Chainer_Mask_R-CNN

#### Light-Head R-CNN

**Light-Head R-CNN: In Defense of Two-Stage Object Detector**

- intro: Tsinghua University & Megvii Inc
- arxiv: <https://arxiv.org/abs/1711.07264>
- github(offical): https://github.com/zengarden/light_head_rcnn
- github: <https://github.com/terrychenism/Deformable-ConvNets/blob/master/rfcn/symbols/resnet_v1_101_rfcn_light.py#L784>

#### Cascade R-CNN

**Cascade R-CNN: Delving into High Quality Object Detection**

- arxiv: <https://arxiv.org/abs/1712.00726>
- github: <https://github.com/zhaoweicai/cascade-rcnn>

#### SPP-Net

**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**

- intro: ECCV 2014 / TPAMI 2015
- arxiv: <http://arxiv.org/abs/1406.4729>
- github: <https://github.com/ShaoqingRen/SPP_net>
- notes: <http://zhangliliang.com/2014/09/13/paper-note-sppnet/>

**DeepID-Net: Deformable Deep Convolutional Neural Networks for Object Detection**

- intro: PAMI 2016
- intro: an extension of R-CNN. box pre-training, cascade on region proposals, deformation layers and context representations
- project page: <http://www.ee.cuhk.edu.hk/%CB%9Cwlouyang/projects/imagenetDeepId/index.html>
- arxiv: <http://arxiv.org/abs/1412.5661>

**Object Detectors Emerge in Deep Scene CNNs**

- intro: ICLR 2015
- arxiv: <http://arxiv.org/abs/1412.6856>
- paper: <https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf>
- paper: <https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf>
- slides: <http://places.csail.mit.edu/slide_iclr2015.pdf>

**segDeepM: Exploiting Segmentation and Context in Deep Neural Networks for Object Detection**

- intro: CVPR 2015
- project(code+data): <https://www.cs.toronto.edu/~yukun/segdeepm.html>
- arxiv: <https://arxiv.org/abs/1502.04275>
- github: <https://github.com/YknZhu/segDeepM>

**Object Detection Networks on Convolutional Feature Maps**

- intro: TPAMI 2015
- keywords: NoC
- arxiv: <http://arxiv.org/abs/1504.06066>

**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction**

- arxiv: <http://arxiv.org/abs/1504.03293>
- slides: <http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf>
- github: <https://github.com/YutingZhang/fgs-obj>

**DeepBox: Learning Objectness with Convolutional Networks**

- keywords: DeepBox
- arxiv: <http://arxiv.org/abs/1505.02146>
- github: <https://github.com/weichengkuo/DeepBox>

#### YOLO

**You Only Look Once: Unified, Real-Time Object Detection**

[![img](https://camo.githubusercontent.com/e69d4118b20a42de4e23b9549f9a6ec6dbbb0814/687474703a2f2f706a7265646469652e636f6d2f6d656469612f66696c65732f6461726b6e65742d626c61636b2d736d616c6c2e706e67)](https://camo.githubusercontent.com/e69d4118b20a42de4e23b9549f9a6ec6dbbb0814/687474703a2f2f706a7265646469652e636f6d2f6d656469612f66696c65732f6461726b6e65742d626c61636b2d736d616c6c2e706e67)

- arxiv: <http://arxiv.org/abs/1506.02640>
- code: <https://pjreddie.com/darknet/yolov1/>
- github: <https://github.com/pjreddie/darknet>
- blog: <https://pjreddie.com/darknet/yolov1/>
- slides: <https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p>
- reddit: <https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/>
- github: <https://github.com/gliese581gg/YOLO_tensorflow>
- github: <https://github.com/xingwangsfu/caffe-yolo>
- github: <https://github.com/frankzhangrui/Darknet-Yolo>
- github: <https://github.com/BriSkyHekun/py-darknet-yolo>
- github: <https://github.com/tommy-qichang/yolo.torch>
- github: <https://github.com/frischzenger/yolo-windows>
- github: <https://github.com/AlexeyAB/yolo-windows>
- github: <https://github.com/nilboy/tensorflow-yolo>

**darkflow - translate darknet to tensorflow. Load trained weights, retrain/fine-tune them using tensorflow, export constant graph def to C++**

- blog: <https://thtrieu.github.io/notes/yolo-tensorflow-graph-buffer-cpp>
- github: <https://github.com/thtrieu/darkflow>

**Start Training YOLO with Our Own Data**

[![img](https://camo.githubusercontent.com/2f99b692dd7ce47d7832385f3e8a6654e680d92a/687474703a2f2f6775616e6768616e2e696e666f2f626c6f672f656e2f77702d636f6e74656e742f75706c6f6164732f323031352f31322f696d616765732d34302e6a7067)](https://camo.githubusercontent.com/2f99b692dd7ce47d7832385f3e8a6654e680d92a/687474703a2f2f6775616e6768616e2e696e666f2f626c6f672f656e2f77702d636f6e74656e742f75706c6f6164732f323031352f31322f696d616765732d34302e6a7067)

- intro: train with customized data and class numbers/labels. Linux / Windows version for darknet.
- blog: <http://guanghan.info/blog/en/my-works/train-yolo/>
- github: <https://github.com/Guanghan/darknet>

**YOLO: Core ML versus MPSNNGraph**

- intro: Tiny YOLO for iOS implemented using CoreML but also using the new MPS graph API.
- blog: <http://machinethink.net/blog/yolo-coreml-versus-mps-graph/>
- github: <https://github.com/hollance/YOLO-CoreML-MPSNNGraph>

**TensorFlow YOLO object detection on Android**

- intro: Real-time object detection on Android using the YOLO network with TensorFlow
- github: <https://github.com/natanielruiz/android-yolo>

**Computer Vision in iOS – Object Detection**

- blog: <https://sriraghu.com/2017/07/12/computer-vision-in-ios-object-detection/>
- github:<https://github.com/r4ghu/iOS-CoreML-Yolo>

#### YOLOv2

**YOLO9000: Better, Faster, Stronger**

- arxiv: <https://arxiv.org/abs/1612.08242>
- code: <http://pjreddie.com/yolo9000/>    https://pjreddie.com/darknet/yolov2/
- github(Chainer): <https://github.com/leetenki/YOLOv2>
- github(Keras): <https://github.com/allanzelener/YAD2K>
- github(PyTorch): <https://github.com/longcw/yolo2-pytorch>
- github(Tensorflow): <https://github.com/hizhangp/yolo_tensorflow>
- github(Windows): <https://github.com/AlexeyAB/darknet>
- github: <https://github.com/choasUp/caffe-yolo9000>
- github: <https://github.com/philipperemy/yolo-9000>
- github(TensorFlow): <https://github.com/KOD-Chen/YOLOv2-Tensorflow>
- github(Keras): <https://github.com/yhcc/yolo2>
- github(Keras): <https://github.com/experiencor/keras-yolo2>
- github(TensorFlow): <https://github.com/WojciechMormul/yolo2>

**darknet_scripts**

- intro: Auxilary scripts to work with (YOLO) darknet deep learning famework. AKA -> How to generate YOLO anchors?
- github: <https://github.com/Jumabek/darknet_scripts>

**Yolo_mark: GUI for marking bounded boxes of objects in images for training Yolo v2**

- github: <https://github.com/AlexeyAB/Yolo_mark>

**LightNet: Bringing pjreddie's DarkNet out of the shadows**

<https://github.com//explosion/lightnet>

**YOLO v2 Bounding Box Tool**

- intro: Bounding box labeler tool to generate the training data in the format YOLO v2 requires.
- github: <https://github.com/Cartucho/yolo-boundingbox-labeler-GUI>

**Loss Rank Mining: A General Hard Example Mining Method for Real-time Detectors**

- intro: **LRM** is the first hard example mining strategy which could fit YOLOv2 perfectly and make it better applied in series of real scenarios where both real-time rates and accurate detection are strongly demanded.
- arxiv: https://arxiv.org/abs/1804.04606

**Object detection at 200 Frames Per Second**

- intro: faster than Tiny-Yolo-v2
- arxiv: https://arxiv.org/abs/1805.06361

**Event-based Convolutional Networks for Object Detection in Neuromorphic Cameras**

- intro: YOLE--Object Detection in Neuromorphic Cameras
- arxiv:https://arxiv.org/abs/1805.07931

**OmniDetector: With Neural Networks to Bounding Boxes**

- intro: a person detector on n fish-eye images of indoor scenes（NIPS 2018）
- arxiv:https://arxiv.org/abs/1805.08503
- datasets:https://gitlab.com/omnidetector/omnidetector

#### YOLOv3

**YOLOv3: An Incremental Improvement**

- arxiv:https://arxiv.org/abs/1804.02767
- paper:https://pjreddie.com/media/files/papers/YOLOv3.pdf
- code: <https://pjreddie.com/darknet/yolo/>
- github(Official):https://github.com/pjreddie/darknet
- github:https://github.com/mystic123/tensorflow-yolo-v3
- github:https://github.com/experiencor/keras-yolo3
- github:https://github.com/qqwweee/keras-yolo3
- github:https://github.com/marvis/pytorch-yolo3
- github:https://github.com/ayooshkathuria/pytorch-yolo-v3
- github:https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
- github:https://github.com/eriklindernoren/PyTorch-YOLOv3
- github:https://github.com/ultralytics/yolov3
- github:https://github.com/BobLiu20/YOLOv3_PyTorch
- github:https://github.com/andy-yun/pytorch-0.4-yolov3
- github:https://github.com/DeNA/PyTorch_YOLOv3

#### YOLT

**You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery**

- intro: Small Object Detection


- arxiv:https://arxiv.org/abs/1805.09512
- github:https://github.com/avanetten/yolt

#### SSD

**SSD: Single Shot MultiBox Detector**

[![img](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)

- intro: ECCV 2016 Oral
- arxiv: <http://arxiv.org/abs/1512.02325>
- paper: <http://www.cs.unc.edu/~wliu/papers/ssd.pdf>
- slides: [http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf)
- github(Official): <https://github.com/weiliu89/caffe/tree/ssd>
- video: <http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973>
- github: <https://github.com/zhreshold/mxnet-ssd>
- github: <https://github.com/zhreshold/mxnet-ssd.cpp>
- github: <https://github.com/rykov8/ssd_keras>
- github: <https://github.com/balancap/SSD-Tensorflow>
- github: <https://github.com/amdegroot/ssd.pytorch>
- github(Caffe): <https://github.com/chuanqi305/MobileNet-SSD>

**What's the diffience in performance between this new code you pushed and the previous code? #327**

<https://github.com/weiliu89/caffe/issues/327>

#### DSSD

**DSSD : Deconvolutional Single Shot Detector**

- intro: UNC Chapel Hill & Amazon Inc
- arxiv: <https://arxiv.org/abs/1701.06659>
- github: <https://github.com/chengyangfu/caffe/tree/dssd>
- github: <https://github.com/MTCloudVision/mxnet-dssd>
- demo: <http://120.52.72.53/www.cs.unc.edu/c3pr90ntc0td/~cyfu/dssd_lalaland.mp4>

**Enhancement of SSD by concatenating feature maps for object detection**

- intro: rainbow SSD (R-SSD)
- arxiv: <https://arxiv.org/abs/1705.09587>

**Context-aware Single-Shot Detector**

- keywords: CSSD, DiCSSD, DeCSSD, effective receptive fields (ERFs), theoretical receptive fields (TRFs)
- arxiv: <https://arxiv.org/abs/1707.08682>

**Feature-Fused SSD: Fast Detection for Small Objects**

<https://arxiv.org/abs/1709.05054>

#### FSSD

**FSSD: Feature Fusion Single Shot Multibox Detector**

<https://arxiv.org/abs/1712.00960>

**Weaving Multi-scale Context for Single Shot Detector**

- intro: WeaveNet
- keywords: fuse multi-scale information
- arxiv: <https://arxiv.org/abs/1712.03149>

#### ESSD

**Extend the shallow part of Single Shot MultiBox Detector via Convolutional Neural Network**

<https://arxiv.org/abs/1801.05918>

**Tiny SSD: A Tiny Single-shot Detection Deep Convolutional Neural Network for Real-time Embedded Object Detection**

<https://arxiv.org/abs/1802.06488>

#### MDSSD

**MDSSD: Multi-scale Deconvolutional Single Shot Detector for small objects**

- arxiv: https://arxiv.org/abs/1805.07009

#### Pelee

**Pelee: A Real-Time Object Detection System on Mobile Devices**

https://github.com/Robert-JunWang/Pelee

- intro: (ICLR 2018 workshop track)


- arxiv: https://arxiv.org/abs/1804.06882
- github: https://github.com/Robert-JunWang/Pelee

#### Fire SSD

**Fire SSD: Wide Fire Modules based Single Shot Detector on Edge Device**

- intro:low cost, fast speed and high mAP on  factor edge computing devices


- arxiv:https://arxiv.org/abs/1806.05363

#### R-FCN

**R-FCN: Object Detection via Region-based Fully Convolutional Networks**

- arxiv: <http://arxiv.org/abs/1605.06409>
- github: <https://github.com/daijifeng001/R-FCN>
- github(MXNet): <https://github.com/msracver/Deformable-ConvNets/tree/master/rfcn>
- github: <https://github.com/Orpine/py-R-FCN>
- github: <https://github.com/PureDiors/pytorch_RFCN>
- github: <https://github.com/bharatsingh430/py-R-FCN-multiGPU>
- github: <https://github.com/xdever/RFCN-tensorflow>

**R-FCN-3000 at 30fps: Decoupling Detection and Classification**

<https://arxiv.org/abs/1712.01802>

**Recycle deep features for better object detection**

- arxiv: <http://arxiv.org/abs/1607.05066>

#### FPN

**Feature Pyramid Networks for Object Detection**

- intro: Facebook AI Research
- arxiv: <https://arxiv.org/abs/1612.03144>

**Action-Driven Object Detection with Top-Down Visual Attentions**

- arxiv: <https://arxiv.org/abs/1612.06704>

**Beyond Skip Connections: Top-Down Modulation for Object Detection**

- intro: CMU & UC Berkeley & Google Research
- arxiv: <https://arxiv.org/abs/1612.06851>

**Wide-Residual-Inception Networks for Real-time Object Detection**

- intro: Inha University
- arxiv: <https://arxiv.org/abs/1702.01243>

**Attentional Network for Visual Object Detection**

- intro: University of Maryland & Mitsubishi Electric Research Laboratories
- arxiv: <https://arxiv.org/abs/1702.01478>

**Learning Chained Deep Features and Classifiers for Cascade in Object Detection**

- keykwords: CC-Net
- intro: chained cascade network (CC-Net). 81.1% mAP on PASCAL VOC 2007
- arxiv: <https://arxiv.org/abs/1702.07054>

**DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling**

- intro: ICCV 2017 (poster)
- arxiv: <https://arxiv.org/abs/1703.10295>

**Discriminative Bimodal Networks for Visual Localization and Detection with Natural Language Queries**

- intro: CVPR 2017
- arxiv: <https://arxiv.org/abs/1704.03944>

**Spatial Memory for Context Reasoning in Object Detection**

- arxiv: <https://arxiv.org/abs/1704.04224>

**Accurate Single Stage Detector Using Recurrent Rolling Convolution**

- intro: CVPR 2017. SenseTime
- keywords: Recurrent Rolling Convolution (RRC)
- arxiv: <https://arxiv.org/abs/1704.05776>
- github: <https://github.com/xiaohaoChen/rrc_detection>

**Deep Occlusion Reasoning for Multi-Camera Multi-Target Detection**

<https://arxiv.org/abs/1704.05775>

**LCDet: Low-Complexity Fully-Convolutional Neural Networks for Object Detection in Embedded Systems**

- intro: Embedded Vision Workshop in CVPR. UC San Diego & Qualcomm Inc
- arxiv: <https://arxiv.org/abs/1705.05922>

**Point Linking Network for Object Detection**

- intro: Point Linking Network (PLN)
- arxiv: <https://arxiv.org/abs/1706.03646>

**Perceptual Generative Adversarial Networks for Small Object Detection**

<https://arxiv.org/abs/1706.05274>

**Few-shot Object Detection**

<https://arxiv.org/abs/1706.08249>

**Yes-Net: An effective Detector Based on Global Information**

<https://arxiv.org/abs/1706.09180>

**SMC Faster R-CNN: Toward a scene-specialized multi-object detector**

<https://arxiv.org/abs/1706.10217>

**Towards lightweight convolutional neural networks for object detection**

<https://arxiv.org/abs/1707.01395>

**RON: Reverse Connection with Objectness Prior Networks for Object Detection**

- intro: CVPR 2017
- arxiv: <https://arxiv.org/abs/1707.01691>
- github: <https://github.com/taokong/RON>

**Mimicking Very Efficient Network for Object Detection**

- intro: CVPR 2017. SenseTime & Beihang University
- paper: <http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Mimicking_Very_Efficient_CVPR_2017_paper.pdf>

**Residual Features and Unified Prediction Network for Single Stage Detection**

<https://arxiv.org/abs/1707.05031>

**Deformable Part-based Fully Convolutional Network for Object Detection**

- intro: BMVC 2017 (oral). Sorbonne Universités & CEDRIC
- arxiv: <https://arxiv.org/abs/1707.06175>

**Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors**

- intro: ICCV 2017
- arxiv: <https://arxiv.org/abs/1707.06399>

**Recurrent Scale Approximation for Object Detection in CNN**

- intro: ICCV 2017
- keywords: Recurrent Scale Approximation (RSA)
- arxiv: <https://arxiv.org/abs/1707.09531>
- github: <https://github.com/sciencefans/RSA-for-object-detection>

#### DSOD

**DSOD: Learning Deeply Supervised Object Detectors from Scratch**

![img](https://user-images.githubusercontent.com/3794909/28934967-718c9302-78b5-11e7-89ee-8b514e53e23c.png)

- intro: ICCV 2017. Fudan University & Tsinghua University & Intel Labs China
- arxiv: <https://arxiv.org/abs/1708.01241>
- github: <https://github.com/szq0214/DSOD>
- github:https://github.com/Windaway/DSOD-Tensorflow
- github:https://github.com/chenyuntc/dsod.pytorch

**Learning Object Detectors from Scratch with Gated Recurrent Feature Pyramids**

- arxiv:https://arxiv.org/abs/1712.00886
- github:https://github.com/szq0214/GRP-DSOD

**Tiny-DSOD: Lightweight Object Detection for Resource-Restricted Usages**

- intro: BMVC 2018
- arXiv: https://arxiv.org/abs/1807.11013

**Object Detection from Scratch with Deep Supervision**

- intro: This is an extended version of DSOD
- arXiv: https://arxiv.org/abs/1809.09294

#### RetinaNet

**Focal Loss for Dense Object Detection**

- intro: ICCV 2017 Best student paper award. Facebook AI Research
- keywords: RetinaNet
- arxiv: <https://arxiv.org/abs/1708.02002>

**CoupleNet: Coupling Global Structure with Local Parts for Object Detection**

- intro: ICCV 2017
- arxiv: <https://arxiv.org/abs/1708.02863>

**Incremental Learning of Object Detectors without Catastrophic Forgetting**

- intro: ICCV 2017. Inria
- arxiv: <https://arxiv.org/abs/1708.06977>

**Zoom Out-and-In Network with Map Attention Decision for Region Proposal and Object Detection**

<https://arxiv.org/abs/1709.04347>

**StairNet: Top-Down Semantic Aggregation for Accurate One Shot Detection**

<https://arxiv.org/abs/1709.05788>

**Dynamic Zoom-in Network for Fast Object Detection in Large Images**

<https://arxiv.org/abs/1711.05187>

**Zero-Annotation Object Detection with Web Knowledge Transfer**

- intro: NTU, Singapore & Amazon
- keywords: multi-instance multi-label domain adaption learning framework
- arxiv: <https://arxiv.org/abs/1711.05954>

#### MegDet

**MegDet: A Large Mini-Batch Object Detector**

- intro: Peking University & Tsinghua University & Megvii Inc
- arxiv: <https://arxiv.org/abs/1711.07240>

**Receptive Field Block Net for Accurate and Fast Object Detection**

- intro: RFBNet
- arxiv: <https://arxiv.org/abs/1711.07767>
- github: <https://github.com//ruinmessi/RFBNet>

**An Analysis of Scale Invariance in Object Detection - SNIP**

- arxiv: <https://arxiv.org/abs/1711.08189>
- github: <https://github.com/bharatsingh430/snip>

**Feature Selective Networks for Object Detection**

<https://arxiv.org/abs/1711.08879>

**Learning a Rotation Invariant Detector with Rotatable Bounding Box**

- arxiv: <https://arxiv.org/abs/1711.09405>
- github: <https://github.com/liulei01/DRBox>

**Scalable Object Detection for Stylized Objects**

- intro: Microsoft AI & Research Munich
- arxiv: <https://arxiv.org/abs/1711.09822>

**Learning Object Detectors from Scratch with Gated Recurrent Feature Pyramids**

- arxiv: <https://arxiv.org/abs/1712.00886>
- github: <https://github.com/szq0214/GRP-DSOD>

**Deep Regionlets for Object Detection**

- keywords: region selection network, gating network
- arxiv: <https://arxiv.org/abs/1712.02408>

**Training and Testing Object Detectors with Virtual Images**

- intro: IEEE/CAA Journal of Automatica Sinica
- arxiv: <https://arxiv.org/abs/1712.08470>

**Large-Scale Object Discovery and Detector Adaptation from Unlabeled Video**

- keywords: object mining, object tracking, unsupervised object discovery by appearance-based clustering, self-supervised detector adaptation
- arxiv: <https://arxiv.org/abs/1712.08832>

**Spot the Difference by Object Detection**

- intro: Tsinghua University & JD Group
- arxiv: <https://arxiv.org/abs/1801.01051>

**Localization-Aware Active Learning for Object Detection**

- arxiv: <https://arxiv.org/abs/1801.05124>

**Object Detection with Mask-based Feature Encoding**

- arxiv: <https://arxiv.org/abs/1802.03934>

**LSTD: A Low-Shot Transfer Detector for Object Detection**

- intro: AAAI 2018
- arxiv: <https://arxiv.org/abs/1803.01529>

**Pseudo Mask Augmented Object Detection**

<https://arxiv.org/abs/1803.05858>

**Revisiting RCNN: On Awakening the Classification Power of Faster RCNN**

<https://arxiv.org/abs/1803.06799>

**Learning Region Features for Object Detection**

- intro: Peking University & MSRA
- arxiv: <https://arxiv.org/abs/1803.07066>

**Single-Shot Bidirectional Pyramid Networks for High-Quality Object Detection**

- intro: Singapore Management University & Zhejiang University
- arxiv: <https://arxiv.org/abs/1803.08208>

**Object Detection for Comics using Manga109 Annotations**

- intro: University of Tokyo & National Institute of Informatics, Japan
- arxiv: <https://arxiv.org/abs/1803.08670>

**Task-Driven Super Resolution: Object Detection in Low-resolution Images**

- arxiv: <https://arxiv.org/abs/1803.11316>

**Transferring Common-Sense Knowledge for Object Detection**

- arxiv: <https://arxiv.org/abs/1804.01077>

**Multi-scale Location-aware Kernel Representation for Object Detection**

- intro: CVPR 2018
- arxiv: <https://arxiv.org/abs/1804.00428>
- github: <https://github.com/Hwang64/MLKP>


**Loss Rank Mining: A General Hard Example Mining Method for Real-time Detectors**

- intro: National University of Defense Technology
- arxiv: https://arxiv.org/abs/1804.04606

**Robust Physical Adversarial Attack on Faster R-CNN Object Detector**

- arxiv: https://arxiv.org/abs/1804.05810

#### RefineNet

**Single-Shot Refinement Neural Network for Object Detection**

- intro: CVPR 2018

- arxiv: <https://arxiv.org/abs/1711.06897>
- github: <https://github.com/sfzhang15/RefineDet>
- github: https://github.com/lzx1413/PytorchSSD
- github: https://github.com/ddlee96/RefineDet_mxnet
- github: https://github.com/MTCloudVision/RefineDet-Mxnet

#### DetNet

**DetNet: A Backbone network for Object Detection**

- intro: Tsinghua University & Face++
- arxiv: https://arxiv.org/abs/1804.06215


#### SSOD

**Self-supervisory Signals for Object Discovery and Detection**

- Google Brain
- arxiv:https://arxiv.org/abs/1806.03370

#### CornerNet

**CornerNet: Detecting Objects as Paired Keypoints**

- intro: ECCV 2018
- arXiv: https://arxiv.org/abs/1808.01244
- github: <https://github.com/umich-vl/CornerNet>

#### M2Det

**M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network**

- intro: AAAI 2019
- arXiv: https://arxiv.org/abs/1811.04533
- github: https://github.com/qijiezhao/M2Det

#### 3D Object Detection

**3D Backbone Network for 3D Object Detection**

- arXiv: https://arxiv.org/abs/1901.08373

**LMNet: Real-time Multiclass Object Detection on CPU using 3D LiDARs**

- arxiv: https://arxiv.org/abs/1805.04902
- github: https://github.com/CPFL/Autoware/tree/feature/cnn_lidar_detection


#### ZSD（Zero-Shot Object Detection）

**Zero-Shot Detection**

- intro: Australian National University
- keywords: YOLO
- arxiv: <https://arxiv.org/abs/1803.07113>

**Zero-Shot Object Detection**

- arxiv: https://arxiv.org/abs/1804.04340

**Zero-Shot Object Detection: Learning to Simultaneously Recognize and Localize Novel Concepts**

- arxiv: https://arxiv.org/abs/1803.06049

**Zero-Shot Object Detection by Hybrid Region Embedding**

- arxiv: https://arxiv.org/abs/1805.06157

#### OSD（One-Shot Object Detection）

**Comparison Network for One-Shot Conditional Object Detection**

- arXiv: https://arxiv.org/abs/1904.02317

**One-Shot Object Detection**

RepMet: Representative-based metric learning for classification and one-shot object detection

- intro: IBM Research AI
- arxiv:https://arxiv.org/abs/1806.04728
- github: TODO

#### Weakly Supervised Object Detection

**Weakly Supervised Object Detection in Artworks**

- intro: ECCV 2018 Workshop Computer Vision for Art Analysis
- arXiv: https://arxiv.org/abs/1810.02569
- Datasets: https://wsoda.telecom-paristech.fr/downloads/dataset/IconArt_v1.zip

**Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation**

- intro: CVPR 2018
- arXiv: https://arxiv.org/abs/1803.11365
- homepage: https://naoto0804.github.io/cross_domain_detection/
- paper: http://openaccess.thecvf.com/content_cvpr_2018/html/Inoue_Cross-Domain_Weakly-Supervised_Object_CVPR_2018_paper.html
- github: https://github.com/naoto0804/cross-domain-detection

#### Softer-NMS

**《Softer-NMS: Rethinking Bounding Box Regression for Accurate Object Detection》**

- intro: CMU & Face++
- arXiv: https://arxiv.org/abs/1809.08545
- github: https://github.com/yihui-he/softer-NMS

**Feature Selective Anchor-Free Module for Single-Shot Object Detection**

- intro: CVPR 2019

- arXiv: https://arxiv.org/abs/1903.00621

**Object Detection based on Region Decomposition and Assembly**

- intro: AAAI 2019

- arXiv: https://arxiv.org/abs/1901.08225

**Bottom-up Object Detection by Grouping Extreme and Center Points**

- intro: one stage 43.2% on COCO test-dev
- arXiv: https://arxiv.org/abs/1901.08043
- github: https://github.com/xingyizhou/ExtremeNet

**ORSIm Detector: A Novel Object Detection Framework in Optical Remote Sensing Imagery Using Spatial-Frequency Channel Features**

- intro: IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING

- arXiv: https://arxiv.org/abs/1901.07925

**Consistent Optimization for Single-Shot Object Detection**

- intro: improves RetinaNet from 39.1 AP to 40.1 AP on COCO datase

- arXiv: https://arxiv.org/abs/1901.06563

**Learning Pairwise Relationship for Multi-object Detection in Crowded Scenes**

- arXiv: https://arxiv.org/abs/1901.03796

**RetinaMask: Learning to predict masks improves state-of-the-art single-shot detection for free**

- arXiv: https://arxiv.org/abs/1901.03353
- github: https://github.com/chengyangfu/retinamask

**Region Proposal by Guided Anchoring**

- intro: CUHK - SenseTime Joint Lab
- arXiv: https://arxiv.org/abs/1901.03278

**Scale-Aware Trident Networks for Object Detection**

- intro: mAP of **48.4** on the COCO dataset
- arXiv: https://arxiv.org/abs/1901.01892


**Large-Scale Object Detection of Images from Network Cameras in Variable Ambient Lighting Conditions**

- arXiv: https://arxiv.org/abs/1812.11901

**Strong-Weak Distribution Alignment for Adaptive Object Detection**

- arXiv: https://arxiv.org/abs/1812.04798

**AutoFocus: Efficient Multi-Scale Inference**

- intro: AutoFocus obtains an **mAP of 47.9%** (68.3% at 50% overlap) on the **COCO test-dev** set while processing **6.4 images per second on a Titan X (Pascal) GPU** 
- arXiv: https://arxiv.org/abs/1812.01600

**NOTE-RCNN: NOise Tolerant Ensemble RCNN for Semi-Supervised Object Detection**

- intro: Google Could
- arXiv: https://arxiv.org/abs/1812.00124

**SPLAT: Semantic Pixel-Level Adaptation Transforms for Detection**

- intro: UC Berkeley
- arXiv: https://arxiv.org/abs/1812.00929

**Grid R-CNN**

- intro: SenseTime
- arXiv: https://arxiv.org/abs/1811.12030

**Deformable ConvNets v2: More Deformable, Better Results**

- intro: Microsoft Research Asia

- arXiv: https://arxiv.org/abs/1811.11168

**Anchor Box Optimization for Object Detection**

- intro: Microsoft Research
- arXiv: https://arxiv.org/abs/1812.00469

**Efficient Coarse-to-Fine Non-Local Module for the Detection of Small Objects**

- intro: https://arxiv.org/abs/1811.12152

**NOTE-RCNN: NOise Tolerant Ensemble RCNN for Semi-Supervised Object Detection**

- arXiv: https://arxiv.org/abs/1812.00124

**Learning RoI Transformer for Detecting Oriented Objects in Aerial Images**

- arXiv: https://arxiv.org/abs/1812.00155

**Integrated Object Detection and Tracking with Tracklet-Conditioned Detection**

- intro: Microsoft Research Asia
- arXiv: https://arxiv.org/abs/1811.11167

**Deep Regionlets: Blended Representation and Deep Learning for Generic Object Detection**

- arXiv: https://arxiv.org/abs/1811.11318

 **Gradient Harmonized Single-stage Detector**

- intro: AAAI 2019
- arXiv: https://arxiv.org/abs/1811.05181

**CFENet: Object Detection with Comprehensive Feature Enhancement Module**

- intro: ACCV 2018
- github: https://github.com/qijiezhao/CFENet

**DeRPN: Taking a further step toward more general object detection**

- intro: AAAI 2019
- arXiv: https://arxiv.org/abs/1811.06700
- github: https://github.com/HCIILAB/DeRPN

**Hybrid Knowledge Routed Modules for Large-scale Object Detection**

- intro: Sun Yat-Sen University & Huawei Noah’s Ark Lab
- arXiv: https://arxiv.org/abs/1810.12681
- github: https://github.com/chanyn/HKRM

**《Receptive Field Block Net for Accurate and Fast Object Detection》**

- intro: ECCV 2018
- arXiv: [https://arxiv.org/abs/1711.07767](https://arxiv.org/abs/1711.07767)
- github: [https://github.com/ruinmessi/RFBNet](https://github.com/ruinmessi/RFBNet)

**Deep Feature Pyramid Reconfiguration for Object Detection**

- intro: ECCV 2018
- arXiv: https://arxiv.org/abs/1808.07993

**Unsupervised Hard Example Mining from Videos for Improved Object Detection**

- intro: ECCV 2018
- arXiv: https://arxiv.org/abs/1808.04285

**Acquisition of Localization Confidence for Accurate Object Detection**

- intro: ECCV 2018
- arXiv: https://arxiv.org/abs/1807.11590
- github: https://github.com/vacancy/PreciseRoIPooling

**Toward Scale-Invariance and Position-Sensitive Region Proposal Networks**

- intro: ECCV 2018
- arXiv: https://arxiv.org/abs/1807.09528

**MetaAnchor: Learning to Detect Objects with Customized Anchors**

- arxiv: https://arxiv.org/abs/1807.00980

**Relation Network for Object Detection**

- intro: CVPR 2018
- arxiv: https://arxiv.org/abs/1711.11575
- github:https://github.com/msracver/Relation-Networks-for-Object-Detection

**Quantization Mimic: Towards Very Tiny CNN for Object Detection**

- Tsinghua University1 & The Chinese University of Hong Kong2 &SenseTime3
- arxiv: https://arxiv.org/abs/1805.02152

**Learning Rich Features for Image Manipulation Detection**

- intro: CVPR 2018 Camera Ready
- arxiv: https://arxiv.org/abs/1805.04953

**SNIPER: Efficient Multi-Scale Training**

- arxiv:https://arxiv.org/abs/1805.09300
- github:https://github.com/mahyarnajibi/SNIPER

**Soft Sampling for Robust Object Detection**

- intro: the robustness of object detection under the presence of missing annotations
- arxiv:https://arxiv.org/abs/1806.06986

**Cost-effective Object Detection: Active Sample Mining with Switchable Selection Criteria**

- intro: TNNLS 2018
- arxiv:https://arxiv.org/abs/1807.00147
- code: http://kezewang.com/codes/ASM_ver1.zip

#### Other

**R3-Net: A Deep Network for Multi-oriented Vehicle Detection in Aerial Images and Videos**

- arxiv: https://arxiv.org/abs/1808.05560
- youtube: https://youtu.be/xCYD-tYudN0

#### Detection Toolbox

- [Detectron(FAIR)](https://github.com/facebookresearch/Detectron): Detectron is Facebook AI Research's software system that implements state-of-the-art object detection algorithms, including [Mask R-CNN](https://arxiv.org/abs/1703.06870). It is written in Python and powered by the [Caffe2](https://github.com/caffe2/caffe2) deep learning framework.
- [Detectron2](https://github.com/facebookresearch/detectron2): Detectron2 is FAIR's next-generation research platform for object detection and segmentation.
- [maskrcnn-benchmark(FAIR)](https://github.com/facebookresearch/maskrcnn-benchmark): Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch.
- [mmdetection(SenseTime&CUHK)](https://github.com/open-mmlab/mmdetection): mmdetection is an open source object detection toolbox based on PyTorch. It is a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).


## Image Segmentation 
⭐⭐⭐
Explore the field of image segmentation, including popular segmentation algorithms, datasets, and cutting-edge approaches for segmenting objects within images.

## Action Classification 
⭐⭐⭐
Discover resources related to action classification, which involves recognizing and classifying human actions in videos. This section includes datasets, models, and techniques specific to action classification tasks.

## Video Recognition 
⭐⭐⭐
In this section, you will find information on video recognition, which involves understanding and analyzing videos to recognize and interpret various visual elements. Explore datasets, models, and techniques for video understanding tasks.

## Deep CNN Models 
⭐⭐⭐
This section is dedicated to deep convolutional neural network (CNN) models, which have revolutionized computer vision research. Discover well-known CNN architectures, pre-trained models, and research papers showcasing the advancements in deep learning for computer vision.

## 3D Vision Models 
⭐⭐⭐
Explore the fascinating world of 3D vision models. This section covers topics such as 3D object recognition, depth estimation, point cloud processing, and related research papers.

## CVML Courses

#### Computer Vision
 * [EENG 512 / CSCI 512 - Computer Vision](http://inside.mines.edu/~whoff/courses/EENG512/) - William Hoff (Colorado School of Mines)
 * [Visual Object and Activity Recognition](https://sites.google.com/site/ucbcs29443/) - Alexei A. Efros and Trevor Darrell (UC Berkeley)
 * [Computer Vision](http://courses.cs.washington.edu/courses/cse455/12wi/) - Steve Seitz (University of Washington)
 * Visual Recognition [Spring 2016](http://vision.cs.utexas.edu/381V-spring2016/), [Fall 2016](http://vision.cs.utexas.edu/381V-fall2016/) - Kristen Grauman (UT Austin)
 * [Language and Vision](http://www.tamaraberg.com/teaching/Spring_15/) - Tamara Berg (UNC Chapel Hill)
 * [Convolutional Neural Networks for Visual Recognition](http://vision.stanford.edu/teaching/cs231n/) - Fei-Fei Li and Andrej Karpathy (Stanford University)
 * [Computer Vision](http://cs.nyu.edu/~fergus/teaching/vision/index.html) - Rob Fergus (NYU)
 * [Computer Vision](https://courses.engr.illinois.edu/cs543/sp2015/) - Derek Hoiem (UIUC)
 * [Computer Vision: Foundations and Applications](http://vision.stanford.edu/teaching/cs131_fall1415/index.html) - Kalanit Grill-Spector and Fei-Fei Li (Stanford University)
 * [High-Level Vision: Behaviors, Neurons and Computational Models](http://vision.stanford.edu/teaching/cs431_spring1314/) - Fei-Fei Li (Stanford University)
 * [Advances in Computer Vision](http://6.869.csail.mit.edu/fa15/) - Antonio Torralba and Bill Freeman (MIT)
 * [Computer Vision](http://www.vision.rwth-aachen.de/course/11/) - Bastian Leibe (RWTH Aachen University)
 * [Computer Vision 2](http://www.vision.rwth-aachen.de/course/9/) - Bastian Leibe (RWTH Aachen University)
 * [Computer Vision](http://klewel.com/conferences/epfl-computer-vision/) Pascal Fua (EPFL):
 * [Computer Vision 1](http://cvlab-dresden.de/courses/computer-vision-1/) Carsten Rother (TU Dresden):
 * [Computer Vision 2](http://cvlab-dresden.de/courses/CV2/) Carsten Rother (TU Dresden):
 * [Multiple View Geometry](https://youtu.be/RDkwklFGMfo?list=PLTBdjV_4f-EJn6udZ34tht9EVIW7lbeo4) Daniel Cremers (TU Munich):




#### Computational Photography
* [Image Manipulation and Computational Photography](http://inst.eecs.berkeley.edu/~cs194-26/fa14/) - Alexei A. Efros (UC Berkeley)
* [Computational Photography](http://graphics.cs.cmu.edu/courses/15-463/2012_fall/463.html) - Alexei A. Efros (CMU)
* [Computational Photography](https://courses.engr.illinois.edu/cs498dh3/) - Derek Hoiem (UIUC)
* [Computational Photography](http://cs.brown.edu/courses/csci1290/) - James Hays (Brown University)
* [Digital & Computational Photography](http://stellar.mit.edu/S/course/6/sp12/6.815/) - Fredo Durand (MIT)
* [Computational Camera and Photography](http://ocw.mit.edu/courses/media-arts-and-sciences/mas-531-computational-camera-and-photography-fall-2009/) - Ramesh Raskar (MIT Media Lab)
* [Computational Photography](https://www.udacity.com/course/computational-photography--ud955) - Irfan Essa (Georgia Tech)
* [Courses in Graphics](http://graphics.stanford.edu/courses/) - Stanford University
* [Computational Photography](http://cs.nyu.edu/~fergus/teaching/comp_photo/index.html) - Rob Fergus (NYU)
* [Introduction to Visual Computing](http://www.cs.toronto.edu/~kyros/courses/320/) - Kyros Kutulakos (University of Toronto)
* [Computational Photography](http://www.cs.toronto.edu/~kyros/courses/2530/) - Kyros Kutulakos (University of Toronto)
* [Computer Vision for Visual Effects](https://www.ecse.rpi.edu/~rjradke/cvfxcourse.html) - Rich Radke (Rensselaer Polytechnic Institute)
* [Introduction to Image Processing](https://www.ecse.rpi.edu/~rjradke/improccourse.html) - Rich Radke (Rensselaer Polytechnic Institute)

#### Machine Learning and Statistical Learning
 * [Machine Learning](https://www.coursera.org/learn/machine-learning) - Andrew Ng (Stanford University)
 * [Learning from Data](https://work.caltech.edu/telecourse.html) - Yaser S. Abu-Mostafa (Caltech)
 * [Statistical Learning](https://class.stanford.edu/courses/HumanitiesandScience/StatLearning/Winter2015/about) - Trevor Hastie and Rob Tibshirani (Stanford University)
 * [Statistical Learning Theory and Applications](http://www.mit.edu/~9.520/fall14/) - Tomaso Poggio, Lorenzo Rosasco, Carlo Ciliberto, Charlie Frogner, Georgios Evangelopoulos, Ben Deen (MIT)
 * [Statistical Learning](http://www.stat.rice.edu/~gallen/stat640.html) - Genevera Allen (Rice University)
 * [Practical Machine Learning](http://www.cs.berkeley.edu/~jordan/courses/294-fall09/) - Michael Jordan (UC Berkeley)
 * [Course on Information Theory, Pattern Recognition, and Neural Networks](http://videolectures.net/course_information_theory_pattern_recognition/) - David MacKay (University of Cambridge)
 * [Methods for Applied Statistics: Unsupervised Learning](http://web.stanford.edu/~lmackey/stats306b/) - Lester Mackey (Stanford)
 * [Machine Learning](http://www.robots.ox.ac.uk/~az/lectures/ml/index.html) - Andrew Zisserman (University of Oxford)
 * [Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120) - Sebastian Thrun (Stanford University)
 * [Machine Learning](https://www.udacity.com/course/machine-learning--ud262) - Charles Isbell, Michael Littman (Georgia Tech)
 * [(Convolutional) Neural Networks for Visual Recognition](https://cs231n.github.io/) - Fei-Fei Li, Andrej Karphaty, Justin Johnson (Stanford University)
 * [Machine Learning for Computer Vision](https://youtu.be/QZmZFeZxEKI?list=PLTBdjV_4f-EIiongKlS9OKrBEp8QR47Wl) - Rudolph Triebel (TU Munich)



#### Optimization
 * [Convex Optimization I](http://stanford.edu/class/ee364a/) - Stephen Boyd (Stanford University)
 * [Convex Optimization II](http://stanford.edu/class/ee364b/) - Stephen Boyd (Stanford University)
 * [Convex Optimization](https://class.stanford.edu/courses/Engineering/CVX101/Winter2014/about) - Stephen Boyd (Stanford University)
 * [Optimization at MIT](http://optimization.mit.edu/classes.php) - (MIT)
 * [Convex Optimization](http://www.stat.cmu.edu/~ryantibs/convexopt/) - Ryan Tibshirani (CMU)


## Additional Resources
⭐⭐⭐
For an extensive list of computer vision resources, you can also check out the [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision) repository by [Jianbo Shi](https://github.com/jbhuang0604). It contains a curated collection of various computer vision topics, including datasets, software, tutorials, and research papers.

## Contributing
⭐⭐⭐
We welcome contributions from the community to make this repository even more comprehensive and up-to-date. If you have any suggestions, please feel free to open an issue or submit a pull request.

## License

This repository is licensed under the [MIT License](LICENSE).
