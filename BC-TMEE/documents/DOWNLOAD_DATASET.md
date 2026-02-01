## Dataset Downloading

### 1. LIBERO 

For LIBERO, you can refer to the [LIBERO source code](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#Datasets).

Alternatively, you can download the datasets locally using the following links:

- [LIBERO-Goal](https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip)
- [LIBERO-Object](https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip)
- [LIBERO-Spatial](https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip)
- [LIBERO-100](https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip)

The structure is:

```
datasets
│
├── libero_10           
│
├── libero_90     
│
├── libero_goal   
│
├── libero_object
│
└── libero_spatial
```

**Attention**: Our experimental setup differs from that of papers like [OpenVLA](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py). In those works, the image observations are saved at a resolution of 256×256 (instead of 128×128) and undergo additional filtering, such as removing "no-op" (zero) actions and unsuccessful demonstrations.
In contrast, our setting uses the raw LIBERO data with lower-resolution images and no filtering.


<!-- #### LIBERO Results -->

<!-- | Method           | Image Encoder | Fuse Module | Policy Head | LIBERO-Goal | LIBERO-Object | LIBERO-Spatial | LIBERO-Long | Avg   |
|:------------------:|:----------------:|:--------------:|:--------------:|:----------------:|:-----------------:|:--------------:|:--------:|:--------:|
| BC-MLP           | ResNet         | MLP          | MLP          | 16.50        | 19.00          | 29.33           | 2.33         | 16.79      |
| **BC-MLP+IB**    | ResNet         | MLP          | MLP          | **27.67**    | **31.50**      | **41.00**       | **2.67**     | **25.71**  |
| BC-RNN           | ResNet         | RNN          | MLP          | 15.17        | 13.33          | 30.67           | 2.33         | 15.38      |
| **BC-RNN+IB**    | ResNet         | RNN          | MLP          | **26.00**    | **17.67**      | **35.17**       | **3.00**     | 20.46      |
| BC-Trans.        | ResNet         | T-Trans.     | MLP          | 67.83        | 41.83          | 68.00           | 15.83        | 48.37      |
| **BC-Trans.+IB** | ResNet         | T-Trans.     | MLP          | **74.17**    | **45.67**      | **72.50**       | **18.00**    | **52.59**  |
| BC-VILT          | S-Trans.       | T-Trans.     | MLP          | 76.17        | 43.00          | 67.17           | 6.50         | 48.21      |
| **BC+VILT+IB**   | S-Trans.       | T-Trans.     | MLP          | **83.83**    | **52.00**      | **70.67**       | **8.67**     | **53.79**  |
| BC-DP            | ResNet         | T-Trans.     | DP Head      | -            | -              | -               | 78.00        | -          |
| **BC-DP+IB**     | ResNet         | T-Trans.     | DP Head      | -            | -              | -               | **84.00**    | -          | -->


### 2. CortexBench

For a detailed introduction to CortexBench, refer to [this link](https://github.com/facebookresearch/eai-vc/tree/main/cortexbench).
In our experiments, we only use four suites from CortexBench: Adroit, DMControl, MetaWorld, and Trifinger.

For dataset download links, see [this link](https://github.com/facebookresearch/eai-vc/blob/main/cortexbench/DATASETS.md).

The structure is:

```
cortexbench
│
├── adroit-expert-v1.0      
│    ├── pen-v0.pickle
│    └── relocate-v0.pickle
│
├── dmc-expert-v1.0     
│    ├── dmc_cheetah_run-v1.pickle
│    ├── ...
│    └── ...
│
├── metaworld-expert-v1.0   
│    ├── assembly-v2-goal-observable.pickle
│    ├── ...
│    └── ...
│
└── trifinger-demos
     ├── move
     │    ├── demo-0000
     │    ├── ...
     │    └── ...
     │
     └── reach
          ├── demo-0000
          ├── ...
          └── ...
```

<!-- #### CortexBench Results -->

<!-- | Method       | Image Encoder | Adroit | Meta-World | DMControl | TriFinger | Avg   |
|:------------:|:--------------:|:---------:|:-----------:|:-----------:|:-----------:|:-----------:|
| *Full Fine-tuning* |
| ResNet       | ResNet         | 66.00  | 81.07       | 74.93      | 71.59      | 73.40  |
| **ResNet+IB**| ResNet         | **72.00** | **83.20**   | **84.94**  | **72.30**  | **78.11** |
| ViT          | ViT            | 35.33  | 31.73       | 10.41      | 55.57      | 33.26  |
| **ViT+IB**   | ViT            | **37.33** | **36.00**   | **12.53**  | **55.93**  | **35.45** |
| *Partial Fine-tuning* |
| R3M          | ViT-S          | 25.33  | 53.07       | 40.31      | 59.87      | 44.65  |
| **R3M+IB**   | ViT-S          | **27.33** | **54.13**   | **41.74**  | **60.63**  | **45.96** |
| Voltron      | ViT-S          | 18.67  | 72.53       | 25.35      | 74.21      | 47.69  |
| **Voltron+IB**| ViT-S         | **21.33** | **74.40**   | **33.16**  | **75.12**  | **51.00** |
| VC-1         | ViT-B          | 24.67  | 77.60       | 53.82      | 72.05      | 57.04  |
| **VC-1+IB**  | ViT-B          | **26.00** | **82.40**   | **54.93**  | **80.13**  | **59.28** |
| MPI          | ViT-S          | 34.67  | 66.40       | 59.45      | 61.91      | 55.61  |
| **MPI+IB**   | ViT-S          | **36.67** | **69.33**   | **61.41**  | **63.34**  | **57.69** | -->
