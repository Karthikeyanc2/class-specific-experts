# Metric Learning Based Class Specific Experts for Open-Set Recognition of Traffic Participants in Urban Areas Using Infrastructure Sensors

## Abstract:

Sensors installed in the infrastructure can make a significant contribution to the advancement of Advanced Driver Assistance Systems (ADAS) and connected mobility. Thermal cameras provide protection against the abuse of personalised data and perform robustly in challenging environmental conditions, making them an excellent choice for infrastructural perception. The goal of this work is to solve the crucial problem of Open-Set Recognition (OSR) for thermal camera-based perception systems installed in the infrastructure. In this paper, a novel modular architecture for OSR called Class Specific Experts (CSE) is proposed, in which, class specialization is achieved using individual
feature spaces. The proposed methodology can be easily embedded in an object detection setting and provides as a main advantage, the possibility of online incremental learning without catastrophic forgetting. This work also introduces a open-source classification dataset called Infrastructure Thermal Dataset (ITD) containing image snippets captured by a thermal camera mounted in the infrastructure. The proposed approach outperforms the compared baselines for the task of OSR on many publicly available thermal and non-thermal datasets, as well as the new ITD dataset.

## Network:

![image](https://github.com/Karthikeyanc2/class-specific-experts/assets/53954194/aa6726e1-d437-4cd4-ba1e-6402c0b1cc3b)

## Dataset:

MNIST / SVHN / CIFAR10 / CIFAR+10/CIFAR+50 : Torch dataset
IMAGENET : bash get_tinyimagenet.sh
FLIR : python3 create_flir2_dataset.py
ITD: unzip OUTDOOR/outdoor.zip

## Train:

python3 train_osr.py --dataset outdoor

## Eval

python3 train_osr.py --dataset outdoor --eval

## Results:

Methods Closed-set accuracy AUROC
FLIR2 ITD FLIR2 ITD
CAC [21] 94.8 97.9 78.5 87.9
RPL - OSRCI [23] 93.9 94.6 79.8 76.7
ARPL - OSRCI [24] 93.8 94.7 80.1 79.3
CSE-A (ours) 95.2 97.8 80.6 88.7
CSE-L (ours) 95.2 97.9 80.8 88.7

## Cite as:

@Article{chandrasekaran2023,
  author  = {Karthikeyan Chandra Sekaran and Lakshman Balasubramanian and Michael Botsch and Wolfgang Utschick},
  journal = {2023 IEEE Intelligent Vehicles Symposium (IV)},
  title   = {Metric Learning Based Class Specific Experts for Open-Set Recognition of Traffic Participants in Urban Areas Using Infrastructure Sensors},
  year    = {2023},
  notes   = {accepted},
}

Part of the code is taken from the following repositories:
https://github.com/dimitymiller/cac-openset
https://github.com/iCGY96/ARPL.git

## Acknowledgement
This work is supported by Bundesministerium für Digitales und Verkehr, Germany under the funding code 45KI05D041.
