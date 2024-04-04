# Applying Co-teaching on NIDS Dataset

This repository contains a PyTorch implementation of the co-teaching and co-teaching+ methods adapted for a Network Intrusion Detection System (NIDS) dataset, inspired by the ICML'19 paper [How does Disagreement Help Generalization against Label Corruption?](https://arxiv.org/abs/1901.04215).

## Introduction

Label noise is a common issue in real-world datasets, which can significantly degrade the performance of deep learning models. The co-teaching strategy involves training two neural networks simultaneously, where each network learns to teach the other network to select and learn from the most reliable samples. This project extends the application of co-teaching to the domain of network intrusion detection, aiming to improve the robustness and generalization of models against label noise in NIDS datasets.

## Requirements

- Python 3.6+
- PyTorch 1.7.0+
- scikit-learn
- imbalanced-learn
- pandas
- numpy
- tqdm

## Dataset

The dataset used in this project is derived from [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), a comprehensive dataset for network intrusion detection. The dataset contains various types of attacks simulated in a testbed to mirror real-world data, alongside benign traffic for a balanced representation.

## Usage

To run the co-teaching+ model on the NIDS dataset, adjust the parameters as needed and execute the following command:

```bash
python main.py --dataset cicids --model_type coteaching_plus --noise_type symmetric --noise_rate 0.2 --seed 1 --num_workers 4 --result_dir results/trial_1/
