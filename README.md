# TTT4Rec
A Test-Time Training Approach for Rapid Adaption in Sequential Recommendation

Zhaoqi Yang, Yanan Wang, and Yong Ge

MIS Department, Eller College of Management, University of Arizona, Tucson, Arizona, USA

## Usage

### Requirements
* Python 3.7+
* PyTorch 1.12+
* CUDA 11.6+
* Install RecBole:
   `pip install recbole`

You can also refer to the required environment specifications in requirements.txt.

### Run
`
python main.py
`

Please specify the dataset in config.yaml. Please set an appropriate maximum sequence length in config.yaml for each dataset before training.


## Paper
You can see the paper at:
http://arxiv.org/abs/2409.19142

## Dataset
The data is available at:
https://drive.google.com/drive/folders/1ugjwxz_QNZqfuDdpzJs2GB7lDBv1Pm31?usp=sharing
