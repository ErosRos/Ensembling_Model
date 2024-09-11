A conformer-based classifier for variable-length utterance processing in anti-spoofing
===============

Implementation of our work "Anti-spoofing Ensembling Model: Dynamic Weight Allocation in Ensemble Models for Improved Voice Biometrics Security" published in Interspeech 2024. For detailed insights into our methodology, you can access the complete paper [here](https://www.isca-archive.org/interspeech_2024/rosello24_interspeech.html#).

## Requirements
First create and activate the environment, and install torch:
```
conda create -n Ensembling_Model python=3.11.5
conda activate Ensembling_Model
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

```
Then install the requirements:
```
pip install -r requirements.txt
```

## Training and evaluation
We used the LA particion of ASVspoof 2019 for training, it can can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336).

We used ASVspoof 2021 database for development and evaluation. LA can be found [here](https://zenodo.org/records/4837263#.YnDIinYzZhE) and DF [here](https://zenodo.org/records/4835108#.YnDIb3YzZhE).

To train and evaluate your own model using the same parameters used in the paper, execute the following command:
```
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 1
```
This will create save the model weights, the scores for LA and DF partitions of ASVspoof 2021 eval and print the obtained EER.


## Citation

If you find this repository valuable for your work, please consider citing our paper:

```
@inproceedings{rosello24_interspeech,
  title     = {Anti-spoofing Ensembling Model: Dynamic Weight Allocation in Ensemble Models for Improved Voice Biometrics Security},
  author    = {Eros Rosello and Angel M. Gomez and Iván López-Espejo and Antonio M. Peinado and Juan M. Martín-Doñas},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {497--501},
  doi       = {10.21437/Interspeech.2024-403},
}
```

Your citation helps us track the impact of our research and acknowledges the contributors' efforts. If you have any questions, feedback, or need further assistance, please feel free to contact me in erosrosello@ugr.es.