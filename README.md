# SoftLMCCL
**[Open-set Recognition of Unseen Macromolecules in Cellular Electron Cryo-Tomograms by Soft Large Margin Centralized Cosine Loss](https://bmvc2019.org/wp-content/uploads/papers/0347-paper.pdf)**

#### Abstract
<br>Cellular  Electron  Cryo-Tomography  (CECT)  is  a  3D  imaging  tool  that  visualizes the structure and spatial organization of macromolecules at sub-molecular resolution ina near native state,  allowing systematic analysis of seen and unseen macromolecules. Methods for high-throughput subtomogram classification on known macromolecules basedon deep learning have been developed.  However, the learned features guided by eitherthe regular Softmax loss or traditional feature descriptors are not well applicable in theopen-set recognition scenarios where the testing data and the training data have a differ-ent label space.  In other words, the testing data contain novel structural classes unseen in the training data.  In this paper, we propose a novel loss function for deep neural net-works to extract discriminative features for unseen macromolecular structure recognitionin CECT, called Soft Large Margin Centralized Cosine Loss (Soft LMCCL). Our SoftLMCCL projects 3D images into a normalized hypersphere that generates features witha large inter-class variance and a low intra-class variance, which can better generalize across data with different classes and in different datasets.  Our experiments on CECT subtomogram recognition tasks using both simulation data and real data demonstrate that we are able to achieve significantly better verification accuracy and reliability compared to classic loss functions. In summary, our Soft LMCCL is a useful design in our detectiontask of unseen structures and is potentially useful in other similar open-set scenarios.
___

**Implementation using PyTorch**
___

### Directory Structure:
```
├── SoftLMCCL
│   ├── loader
│   │   ├── __init__.py
│   │   ├── dataset.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── ResNet.py
│   ├── loss
│   │   ├── __init__.py
│   │   ├── loss.py
│   ├── model.py
│   ├── LICENSE
│   ├── README.md
│   ├── check_split.py
│   ├── csv_split.py
│   ├── requirements.txt
│   ├── test.py
│   ├── test_req.py
│   ├── train.py
│   └── .gitignore
└── data_
    ├── data1_SNR003
    ├── data2_SNR005
    └── data3_SNRinf
```

### Dependencies:
- For installing dependencies, run:
```
pip install -r SoftLMCCL/requirements.txt
```
- To check packages of your system and corresponding versions, run:
```
python3 SoftLMCCL/test_req.py
```
### Splitting Data:
- Before training split the data into training, testing, and validation sets. Run:
```
python3 SoftLMCCL/csv_split.py
```
- New-directory:
```
└── csv_split
    ├── test.csv
    ├── train.csv
    └── valid.csv
```
- To check split nature and classes, Run:
```
python3 SoftLMCCL/check_split.py
```
- *Note: Classes have been assigned on the basis of the subtomogram filenames as I had data corresponding to that. Depending on your data, you might need to change the dataloader and the splitter. In this case, the main files to be modified are ```SoftLMCCL/csv_splt.py``` and ```SoftLMCCL/loader/dataset.py```*

### Training
- To train the model, run:
```
python3 SoftLMCCL/train.py
```
- New Directory:
```
└── model_save
     ├── checkpoint0.pt
     ├── checkpoint1.pt
     ├── ...
     └── loss.txt
```
- The model generates feature maps which can be later on separated with a simple linear layers. We do not have any accuracy metrics at this point because no predictions are being made -- only features from subtomograms are being extracted.
- *Note: The model uses a lot of overhead and needs GPU support in order to execute. For this, you should have the latest NVIDIA drivers and PyTorch compiled with CUDA*

### Testing/Validation
- To generate predictions and to classify from generated features, run:
```
python3 SoftLMCCL/test.py model_save/checkpoint.pt
```
### Results:
For reference, the results to the validation split of the data on which the model was originally trained is provided below:
<br>
<br>
![Report](report.png?raw=true "Classification Report")

##### Code for paper released as part of work during Research Internship at the Xu lab of Computational Biology, Carnegie Mellon University.
Supervisors: [Sinuo Liu](https://scholar.google.com/citations?user=EVMvLssAAAAJ&hl=zh-CN), [Dr Min Xu](https://scholar.google.com/citations?user=Y3Cqt0cAAAAJ&hl=en)


