# pothole-detection-project/predict.py
import torch
import torch.nn as nn
from torchvision import transforms
import yaml
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models.pothole_detector import PotholeDetector

def main():
    parser = argparse.ArgumentParser(description='Pothole Detection Prediction')
    parser.add_argument('--image_path', type=str,