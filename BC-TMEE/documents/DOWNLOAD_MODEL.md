# Dataset Downloading

## 1. LIBERO Experiments
In LIBERO experiments, there is a pre-trained language model: [bert-base-cased](https://huggingface.co/google-bert/bert-base-cased/tree/main). We recommend downloading it manually beforehand.


## 2. CortexBench Experiments

In CortexBench experiments, only partial fine-tuning methods require loading pre-trained models, specifically [R3M](https://github.com/facebookresearch/r3m), [MVP](https://github.com/ir413/mvp), [VC-1](https://github.com/facebookresearch/eai-vc), [Voltron](https://github.com/siddk/voltron-robotics), and [MPI](https://github.com/OpenDriveLab/MPI). 
**Notably, for the R3M and MVP pre-trained models, we use the reimplementation of Voltron.**

For a better understanding of these methods, refer to the [relevant tutorials](notebooks/methods). Of course, referring to the original code is the best approach!

---

###  Downloading Automatically

If you want the code to automatically download the required files during execution, set `load_path` to `None`, as shown below:

```bash
if cfg.policy.embedding in ['r3m-rn50', 'r3m-small']:
    load_path = os.path.join(cfg.policy.embedding_dir, 'r3m', cfg.policy.embedding) # -> None
    self.feature_extractor = load_r3m("r-r3m-vit", load_path=load_path, only_return_model=True)
    if cfg.train.ft_method == 'partial_ft':
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    self.vector_extractor = instantiate_extractor(self.feature_extractor)()
else:
    raise ValueError("R3M model type is wrong! The repo only suits for [\"r3m-rn50\" and \"r3m-small\"].")
```

We recommend downloading these models into the same folder and setting `load_path` accordingly. 
The directory structure should be as follows:

```bash
models
│
├── bert-base-cased
│
├── distilbert-base-uncased
│
├── mpi    
│    └── mpi-small
│         ├── MPI-small-state_dict.pt
│         └── MPI-small.json 
│    
├── mvp    
│    └── mvp-small 
│         ├── r-mvp.json
│         └── r-mvp.pt 
│
├── r3m    
│    └── mvp-small 
│         ├── r-r3m-vit.json
│         └── r-r3m-vit.pt 
├── vc-1   
│    └── vc1_vitb.pth
│        
└── voltron    
     └── v-cond-small 
          ├── v-cond.json
          └── v-cond.pt    
```

### Downloading Manually

We have also summarized the checkpoint links for all methods from the source code, as follows:  

| Method                                                    |    Github   |    Model    |  
|-----------------------------------------------------------|:-----------:|:-----------:|
| [R3M](https://arxiv.org/abs/2203.12601)                   |   [link](https://github.com/facebookresearch/r3m)    |  ViT-S \[ [checkpoint](https://drive.google.com/file/d/1Yby5oB4oPc33IDQqYxwYjQV3-56hjCTW/view?usp=sharing)  \|  [conifg](https://drive.google.com/file/d/1JGk32BLXwI79uDLAGcpbw0PiupBknf-7/view?usp=sharing)  \]   |
| [MVP](https://arxiv.org/abs/2210.03109)                   |   [link](https://github.com/ir413/mvp)               |  ViT-S \[ [checkpoint](https://drive.google.com/file/d/1-ExshZ6EC8guElOv_s-e8gOJ0R1QEAfj/view?usp=sharing)  \|  [conifg](https://drive.google.com/file/d/1KKNWag6aS1xkUiUjaJ1Khm9D6F3ROhCR/view?usp=sharing)  \]         | 
| [VC-1](https://arxiv.org/abs/2303.18240)                  |   [link](https://github.com/facebookresearch/eai-vc) |  ViT-B \[ [checkpoint](https://dl.fbaipublicfiles.com/eai-vc/vc1_vitb.pth) \]       |
| [Voltron](https://arxiv.org/abs/2302.12766)               |   [link](https://github.com/siddk/voltron-robotics)  |  ViT-S \[ [checkpoint](https://drive.google.com/file/d/12g5QckQSMKqrfr4lFY3UPdy7oLw4APpG/view?usp=sharing)  \|  [conifg](https://drive.google.com/file/d/1O4oqRIblfS6PdFlZzUcYIX-Rqe6LbvnD/view?usp=sharing)  \] | 
| [MPI](https://www.arxiv.org/abs/2406.00439)               |   [link](https://github.com/OpenDriveLab/MPI)        |  ViT-S \[ [checkpoint](https://drive.google.com/file/d/1N7zCWi9ztrcCHsm4xhAA1hsnviv2gdvn/view)  \|  [conifg](https://drive.google.com/file/d/1zG9O9-F86hJowxCUrgpVFfanbIjwn9Tp/view)  \]       |   

Additionally, the MPI initialization model requires the language model: [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased/tree/main). We recommend downloading it manually beforehand.




