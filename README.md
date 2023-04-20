# DenseNet-with-AFF-module
![](./architecture.jpg)

[Attentional feature fusion](https://arxiv.org/abs/2009.14082)

[ICDAR 2013 Chinese Handwriting Recognition Competition](https://ieeexplore.ieee.org/abstract/document/6628856)

What's in this repo so far
- code for training ICDAR dataset
- code for DenseNet with AFF module

## Experiments

### ICDAR2013 Test Datasets
| Architecture                | Top1 Accuracy(%) |
| -------------------         |:-------------:   |
| Fujitsu[[1]](./README.md#References)                     | 94.77            | 
| IDSIAnn                     | 94.42            |
| MCDNN                       | 95.79            |  
| MCDNN                       | 95.79            |  
| GoogleNet-ResNet            | 97.03            | 
| Google-ResNet+triplet loss  | 97.03            | 
| ResNet+Center loss          | 97.03            | 
| DenseNet+AFF(ours)          | 97.23            |

## References
[1] Yin, F., Wang, Q. F., Zhang, X. Y., & Liu, C. L. (2013, August). ICDAR 2013 Chinese handwriting recognition competition. 
In 2013 12th international conference on document analysis and recognition (pp. 1464-1470). IEEE.
