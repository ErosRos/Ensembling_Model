import torch 
import random
import os
import numpy as np
import pandas as pd

def reproducibility(random_seed, args=None):                                  
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn_deterministic = True
    cudnn_benchmark = False
    print("cudnn_deterministic set to False")
    print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return

def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def leer_columnas(archivo_csv, c_names, c_labels, c_phase, phase):
    try:
        df = pd.read_csv(archivo_csv, sep=" ", header=None)
        if phase:
            df = df[[c_names, c_labels, c_phase]]
            df.columns = ["ID", "Label", "Phase"]
        else:
            df = df[[c_names, c_labels]]
            df.columns = ["ID", "Label"]

        return df
    except FileNotFoundError:
        print("El archivo de metadatos no existe")
    except KeyError:
        print("Las columnas especificadas no existen en el archivo de metadatos")


def merge(score_file, metadata='Metadata/LA/trial_metadata.txt', col_names=1, col_labels=5, phase='eval', col_phase=7):
    metadatos = leer_columnas(metadata, col_names, col_labels, col_phase, phase)
    scores = pd.read_csv(score_file, sep=" ", header=None, skipinitialspace=True)
    scores.columns = ["ID", "Scores"]
    if len(scores) != len(metadatos):
        print("CHECK: submission has %d of %d expected trials." % (len(scores), len(metadatos)))
    if len(scores.columns) > 2:
        print("CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces."
              % len(scores.columns)
              )
        exit(1)
    if phase:
        cm_scores = scores.merge(metadatos[metadatos["Phase"] == phase], on="ID")
    else:
        cm_scores = scores.merge(metadatos, on="ID")
    return cm_scores

def extract_eer(score_file, metadata='Metadata/ASVspoof2021_LA_eval/trial_metadata.txt', col_names=1, col_labels=5, phase='eval', col_phase=7):
    cm_scores= merge(score_file, metadata, col_names, col_labels, phase, col_phase)
    bona_cm = cm_scores[cm_scores["Label"] == "bonafide"]["Scores"].values
    spoof_cm = cm_scores[cm_scores["Label"] == "spoof"]["Scores"].values
    eer_cm, threshold = compute_eer(bona_cm, spoof_cm)
    out_data = "EER: {:.4f}. Threshold: {:.5f}\n".format(100 * eer_cm, threshold)
    print(out_data)
    return eer_cm, threshold