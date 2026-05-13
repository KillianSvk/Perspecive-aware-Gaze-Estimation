import random
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_ROOT = Path('data_sets/MPIIFaceGaze')

def load_calibration(participant_dir: Path):
    mat_cam = scipy.io.loadmat(str(participant_dir / 'Calibration' / 'Camera.mat'))
    mat_monitor = scipy.io.loadmat(str(participant_dir / 'Calibration' / 'monitorPose.mat'))
    return mat_cam, mat_monitor

def calculate_vanishing_points():
    pass

def visualize_vanishing_points(participant: str):
    participant_dir = DATASET_ROOT / participant
    mat_cam, mat_monitor = load_calibration(participant_dir)

    K = mat_cam['cameraMatrix']
    # R = mat_monitor['rvecs']

    print(K)

if __name__ == "__main__":
    visualize_vanishing_points('p01')