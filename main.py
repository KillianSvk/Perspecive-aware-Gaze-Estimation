import random
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_ROOT = Path('data_sets/MPIIFaceGaze')


def print_calibration():
    for participant_dir in sorted(DATASET_ROOT.iterdir()):
        cam_file = participant_dir / 'Calibration' / 'Camera.mat'
        if not cam_file.exists():
            continue

        mat = scipy.io.loadmat(cam_file)
        K = mat['cameraMatrix']          # (3, 3)
        d = mat['distCoeffs'].flatten()  # [k1, k2, p1, p2, k3]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        k1, k2, p1, p2, k3 = d[0], d[1], d[2], d[3], d[4]

        print(f"{participant_dir.name}:")
        print(f"  Focal length : fx={fx:.2f}, fy={fy:.2f}")
        print(f"  Principal pt : cx={cx:.2f}, cy={cy:.2f}")
        print(f"  Distortion   : k1={k1:.6f}, k2={k2:.6f}, p1={p1:.6f}, p2={p2:.6f}, k3={k3:.6f}")
        print()

def compute_vanishing_points(mat_cam: dict, mat_monitor: dict) -> dict:
    K    = mat_cam['cameraMatrix']                   # (3, 3)
    rvec = mat_monitor['rvects'].flatten()           # rotation vector (3,)

    R, _ = cv2.Rodrigues(rvec)                       # (3, 3) monitor→camera rotation

    def project_direction(d: np.ndarray) -> tuple:
        """Project a 3D direction vector to a 2D vanishing point in pixels."""
        p = K @ (R @ d)
        return (p[0] / p[2], p[1] / p[2])

    return {
        'horizontal': project_direction(np.array([1.0, 0.0, 0.0])),  # monitor X axis
        'vertical':   project_direction(np.array([0.0, 1.0, 0.0])),  # monitor Y axis
        'normal':     project_direction(np.array([0.0, 0.0, 1.0])),  # monitor normal
    }


if __name__ == "__main__":
    mat_cam     = scipy.io.loadmat('data_sets/MPIIFaceGaze/p00/Calibration/Camera.mat')
    mat_monitor = scipy.io.loadmat('data_sets/MPIIFaceGaze/p00/Calibration/monitorPose.mat')

    vps = compute_vanishing_points(mat_cam, mat_monitor)
    for name, (x, y) in vps.items():
        print(f"  {name:12s}: ({x:.1f}, {y:.1f}) px")