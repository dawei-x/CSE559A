# CSE559A


### Overview

This repository contains partial code files related to our replication project **Conformal Prediction for Image Classification** as part of `CSE 559A: Computer Vision` taught by [Dr. Nathan Jacobs](https://engineering.wustl.edu/faculty/Nathan-Jacobs.html). Full project details are not disclosed here due to privacy and academic integrity constraints as a student group project in WashU CSE. 

We appreciate the authors' official repository [conformal_classification](https://github.com/aangelopoulos/conformal_classification), from which we largely refactored our code. Our effort was intended to replicate experiments in the authors' paper 
[Uncertainty Sets for Image Classifiers using Conformal Prediction](https://arxiv.org/abs/2009.14193)
```
@article{angelopoulos2020sets,
  title={Uncertainty Sets for Image Classifiers using Conformal Prediction},
  author={Angelopoulos, Anastasios N and Bates, Stephen and Malik, Jitendra and Jordan, Michael I},
  journal={arXiv preprint arXiv:2009.14193},
  year={2020}
}
```


### For Colab:
We ran all our code entirely on Colab, with [exp_v0.py](./exp_v0.py) merging all needed functions to replicate experiments. Please note that some components in original implementation/functions have been removed for the sake of concision. Refer to [Exp3.ipynb](./Exp3.ipynb) for a general pipeline and use cases. 
### Prep:
Within your Google Drive's Colab Notebooks directory, create a folder named `cse559a` and upload [exp_v0.py](./exp_v0.py) into that folder. It would be smoother to manually download [imagenet_val](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) to `cse559a` and rename it as `imagenet_val.tar`. 

### Reference
[Loading ImageNet](https://colab.research.google.com/drive/1EBz4feoaUvz-o_yeMI27LEQBkvrXNc_4?usp=sharing)<br/>
[Dataset Preparation](https://github.com/pytorch/examples/issues/275)
