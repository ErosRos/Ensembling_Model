from torch.utils.data import Dataset
import os
import pandas as pd
import soundfile as sf
import numpy as np
import torch


def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

def get_lists(protocols, column_names=1, is_eval=False, column_labels=4):
    data = protocols
    names = data[column_names].to_list()
    if is_eval:
        return names
    else:
        labels = [0 if label == 'bonafide' else 1 for label in data[column_labels].to_list()]
        return names, labels

class fusion_dataset_training(Dataset):
    def __init__(
            self,
            dir_databases,
            name_db,
            track,
            sub_set,
            file_format,
            dir_protocols,
            column_names,
            column_labels,
            scores_folder,
            norm
    ):
        if sub_set == "train":
            end = ".trn.txt"
        else:
            end = ".trl.txt"
        protocols = os.path.join(dir_protocols, track, name_db + "." + track + ".cm." + sub_set + end)
        self.dir_db = os.path.join(dir_databases, track, name_db + "_" + track + "_" + sub_set, file_format)
        self.file_format = file_format

        self.protocols_df = pd.read_csv(protocols, sep=" ", header=None)
        if name_db=='ASVspoof2021': # If we use 2021 for training or validation we only use progress phase.
            self.protocols_df = self.protocols_df[self.protocols_df[7] == 'progress']

        self.list_ids, self.list_labels = get_lists(self.protocols_df,
                                                    column_labels=column_labels,
                                                    column_names=column_names,
                                                    is_eval=False
                                                    )
        
        end = '.txt'
        if norm: end='_norm.txt'
        if scores_folder:
            scores_folder=os.path.join(scores_folder, track, name_db + "_" + track + "_" + sub_set)
            model1 = os.path.join(scores_folder, 'CQCC_GMM' + end)
            model2 = os.path.join(scores_folder, 'LFCC_GMM' + end)
            model3 = os.path.join(scores_folder, 'LFCC_LCNN' + end)
            model4 = os.path.join(scores_folder, 'RawNet2' + end)
            self.score1 = pd.read_csv(model1, sep=" ", header=None).set_index(0)
            self.score2 = pd.read_csv(model2, sep=" ", header=None).set_index(0)
            self.score3 = pd.read_csv(model3, sep=" ", header=None).set_index(0)
            self.score4 = pd.read_csv(model4, sep=" ", header=None).set_index(0)
            self.score = True
        else: self.score = False

    def _read_sample(self, file_id):
        file = os.path.join(self.dir_db, str(file_id) + "." + self.file_format)
        raw_waveform, _ = sf.read(file)
        return raw_waveform

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, idx):
        file_id = self.list_ids[idx]
        # Get waveform
        y = self._read_sample(file_id)
        y = pad(y, max_len=66800)
        # Get label
        label = self.list_labels[idx]
        line = list(self.protocols_df.iloc[idx])
        
        if self.score:
            scores=[self.score1.loc[file_id, 1], self.score2.loc[file_id, 1], self.score3.loc[file_id, 1], self.score4.loc[file_id, 1]]
        else: scores=1
        sample = (file_id, y, label, torch.Tensor(scores))
        return sample


class fusion_dataset_eval(Dataset):
    def __init__(
            self,
            dir_databases,
            name_db,
            track,
            sub_set,
            file_format,
            dir_protocols,
            column_names_eval,
            scores_folder,
            norm
    ):
        """
        """
        if sub_set == "train":
            end = ".trn.txt"
        else:
            end = ".trl.txt"
        protocols = os.path.join(dir_protocols, track, name_db + "." + track + ".cm." + sub_set + end)
        protocols_df = pd.read_csv(protocols, sep=" ", header=None)

        self.dir_db = os.path.join(dir_databases, track, name_db + "_" + track + "_" + sub_set, file_format)
        self.file_format = file_format
        self.list_ids = get_lists(protocols_df, column_names=column_names_eval, is_eval=True)

        scores_folder=os.path.join(scores_folder, track, name_db + "_" + track + "_" + sub_set)
        end = '.txt'
        if norm: end='_norm.txt'
        model1 = os.path.join(scores_folder, 'CQCC_GMM' + end)
        model2 = os.path.join(scores_folder, 'LFCC_GMM' + end)
        model3 = os.path.join(scores_folder, 'LFCC_LCNN' + end)
        model4 = os.path.join(scores_folder, 'RawNet2' + end)
        self.score1 = pd.read_csv(model1, sep=" ", header=None).set_index(0)
        self.score2 = pd.read_csv(model2, sep=" ", header=None).set_index(0)
        self.score3 = pd.read_csv(model3, sep=" ", header=None).set_index(0)
        self.score4 = pd.read_csv(model4, sep=" ", header=None).set_index(0)

    def _read_sample(self, file_id):
        file = os.path.join(self.dir_db, str(file_id) + "." + self.file_format)
        raw_waveform, _ = sf.read(file)
        return raw_waveform

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, idx):
        file_id = self.list_ids[idx]
        # Get noisy file
        waveform = self._read_sample(file_id)
        waveform = pad(waveform, max_len=66800)
        # Get clean file
        scores=[self.score1.loc[file_id, 1], self.score2.loc[file_id, 1], self.score3.loc[file_id, 1], self.score4.loc[file_id, 1]]

        sample = (file_id, waveform, torch.Tensor(scores))
        return sample