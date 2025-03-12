# GLA Transformer
My attempt at implementing the [Gated Linear Attention Transformer](https://www.alphaxiv.org/abs/2312.06635) architecture. The repo has the following structure:  
```
project
│   README.md
│   train.py
|   requirements.txt
│
└───models
│   │   attention.py
│   │   norms.py
|   |   transformer.py
│   
└───utils
|   │   checkpoint.py
|   │   logger.py
|
└───data
|   |   dataset.py
|   |
```  
In short, you should refer to `requirements.txt` if a versions mismatch is encountered:
```bash
pip install -r requirements.txt
```
Then, you can start the training loop by running
```bash
python train.py
```
The training loop logs are automatically saved to the `training.log` file.  

You can modify the training config by changing the `main.config` dict variable values in `train.py`, which were hard-coded there for simplicity.  

`models/attention.py` contains the implementation of GLA.  
`models/transformer.py` contains the implementation of the GLA transformer.  
`models/norms.py` contains the implementations of RMSNorm, SwiGLU.  

By default, both training and validation datasets are downloaded at the start of the training loop as small samples of the C4 dataset. For your custom data loading, modify the `data/dataset.py` file.

Small utility scripts, like the logging module setup, are stored in the `utils` folder of the repo.  

Python version: 3.13.2