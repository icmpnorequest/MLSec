# MLSec
It's a repository to implement some experiments on Machine Learning Security
<br> </br>

## Code Description

### 1. Data_Poisoning Directory

- label_poisoning_attack_svm.py

It aims to implement a data poisoning attack on labels with SVM classifier.


### 2. Membership_Inference Directory

The code under this directory is implementation of 17 S&P paper "Membership Inference Attacks Against Machine Learning Models" by Shokri et al.

- data_synthesis.py

It aims to implement `Algorithm 1: Data Synthesis` in paper.

- shadow_models.py

It aims to implement `Shadow model technique` in paper.

- attack_models.py

It aims to implement `Membership Inference Attack` according to paper.

- Experiment_CIFAR10.py

It aims to implement CIFAR-10 experiment in paper.

- nn_model.py

It completes the neural network code using Pytorch.

- utils.py

It completes the utils code.

\# Please notice: the norm_all_batch_data.npy under the directory `Membership_Inference/cifar10/norm_all_batch_data.npy`
is too large to upload.

I uploaded it to the link: https://pan.baidu.com/s/1uZaZhVYUiRXi3resfuJoiA


## Reference
### 1. Paper

- [17 S&P] Membership Inference Attacks Against Machine Learning Models

- [15 IJSN Attribute Infenrence Attack] Hacking Smart Machines with Smarter Ones How to Extract Meaningful Data from Machine Learning Classifiers


### 2. Code

**Membership Inference Attack**: 

- Paper code: https://github.com/csong27/membership-inference

- BielStela's experiment implementation: https://github.com/BielStela/membership_inference