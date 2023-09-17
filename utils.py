import librosa
import matplotlib.pyplot as plt
import torch
import random
import scipy.io.wavfile as wavfile
import argparse
import os
import csv
import numpy as np
import cv2
from scipy import stats
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from fastdtw import fastdtw
from scipy.signal import correlate
from pystoi.stoi import stoi
from scipy.interpolate import interp2d

def get_correct_classified_data(test_loader, model, device, nums_data):
    correct_datas = []
    correct_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            scores = model(data)
            _, pre = torch.max(scores, dim=1)
            correct_mask = pre.eq(label)
            correct_datas.append(data[correct_mask])
            correct_labels.append(label[correct_mask])
        correct_data = torch.cat(correct_datas)
        correct_label = torch.cat(correct_labels)
    indices = random.sample(range(correct_data.shape[0]), k=nums_data)
    correct_data = correct_data[indices]
    correct_label = correct_label[indices]
    return correct_data, correct_label


def calculate_snr(original_specs, noisy_specs):
    noise_specs = original_specs - noisy_specs
    signal_power = torch.mean(original_specs ** 2)
    noise_power = torch.mean(noise_specs ** 2)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def convert_mel_to_audio(mel_spec_db, file_path):
    mel_spec_power = librosa.db_to_power(mel_spec_db, ref=1.0)
    audio_reconstructed = librosa.feature.inverse.mel_to_audio(mel_spec_power, sr=22050, n_fft=2048, hop_length=512)
    audio_reconstructed = librosa.util.normalize(audio_reconstructed)
    wavfile.write(file_path, rate=22050, data=audio_reconstructed)


def create_attack_parser(model, attack, nums_data):
    parser = argparse.ArgumentParser(description="Create attack configurations")
    parser.add_argument('--model', type=str, default=model, help='The model to be attacked')
    parser.add_argument('--attack', type=str, default=attack, help='Attack type setting')
    parser.add_argument('--nums_data', type=int, default=nums_data, help='Number of attacked data')
    arg = parser.parse_args()

    return arg

def average_euclidean_distance(tensor1, tensor2):
    def euclidean_distance(arr1, arr2):
        return torch.sqrt(torch.sum((arr1 - arr2) ** 2))

    total_distance = 0.0
    num_samples = tensor1.shape[0]

    for i in range(num_samples):
        distance = euclidean_distance(tensor1[i], tensor2[i])
        total_distance += distance

    average_distance = total_distance / num_samples
    return average_distance.item()


def calculate_average_manhattan_distance(tensor1, tensor2):
    manhattan_distances = torch.abs(tensor1 - tensor2).sum(dim=(2, 3))
    mean_manhattan_distances = manhattan_distances.mean(dim=1)
    average_mean_manhattan_distance = mean_manhattan_distances.mean()
    return average_mean_manhattan_distance.item()

def calculate_mean_cosine_similarity(tensor1, tensor2):
    flat_tensor1 = tensor1.view(tensor1.size(0), -1)
    flat_tensor2 = tensor2.view(tensor2.size(0), -1)
    cos_similarities = F.cosine_similarity(flat_tensor1, flat_tensor2, dim=1, eps=1e-8)
    mean_cos_similarity = torch.mean(cos_similarities)
    return mean_cos_similarity.item()

def calculate_average_pearson(tensor1, tensor2):
    tensor1_cpu = tensor1.clone().cpu()
    tensor2_cpu = tensor2.clone().cpu()
    tensor1_flat = tensor1_cpu.view(tensor1.size(0), -1)
    tensor2_flat = tensor2_cpu.view(tensor2.size(0), -1)
    correlation_coefficients = []
    for i in range(tensor1.size(0)):
        corr = np.corrcoef(tensor1_flat[i], tensor2_flat[i])[0, 1]
        correlation_coefficients.append(corr)
    average_correlation = np.mean(correlation_coefficients)
    return average_correlation

def calculate_psnr(tensor1, tensor2):
    mse = F.mse_loss(tensor1, tensor2)
    max_value = torch.max(tensor1)
    psnr = 10 * torch.log10((max_value ** 2) / mse)
    return psnr.item()

def calculate_average_psnr(tensor1, tensor2):
    total_psnr = 0.0
    num_samples = tensor1.size(0)
    for i in range(num_samples):
        mel1 = tensor1[i].cpu()
        mel2 = tensor2[i].cpu()
        psnr = calculate_psnr(mel1, mel2)
        total_psnr += psnr
    average_psnr = total_psnr / num_samples
    return average_psnr


def calculate_mean_ssim(tensor1, tensor2):
    tensor1_clone = tensor1.clone()
    tensor2_clone = tensor2.clone()

    array1 = tensor1_clone.cpu().numpy()
    array2 = tensor2_clone.cpu().numpy()
    ssim_values = []
    for i in range(array1.shape[0]):
        img_ssim = ssim(array1[i, 0], array2[i, 0], data_range=array1.max() - array1.min())
        ssim_values.append(img_ssim)

    ssim_array = np.array(ssim_values)
    mean_ssim = np.mean(ssim_array)
    return mean_ssim