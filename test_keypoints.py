"""Feasibility probe: run RF-DETR keypoint preview on stylized foxgirl renders
and score T-pose adherence from shoulder/elbow/wrist geometry."""
import math
import sys

import numpy as np
from PIL import Image

COCO_KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
           "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
           "left_wrist", "right_wrist", "left_hip", "right_hip",
           "left_knee", "right_knee", "left_ankle", "right_ankle"]


def arm_angle(shoulder, wrist):
    """Angle of the arm vs horizontal, degrees (0 = perfectly horizontal)."""
    dx, dy = wrist[0] - shoulder[0], wrist[1] - shoulder[1]
    return abs(math.degrees(math.atan2(dy, abs(dx))))


def main(paths):
    from rfdetr import RFDETRKeypointPreview
    model = RFDETRKeypointPreview()
    for path in paths:
        img = Image.open(path).convert("RGB")
        det = model.predict(img, threshold=0.4)  # supervision KeyPoints
        n = len(det)
        if n == 0:
            print(f"{path}: NO PERSON DETECTED")
            continue
        kps = det.xy[0]
        conf = det.keypoint_confidence[0] if det.keypoint_confidence is not None else np.ones(len(kps))
        idx = {name: i for i, name in enumerate(COCO_KP)}
        ls, rs = kps[idx["left_shoulder"]], kps[idx["right_shoulder"]]
        lw, rw = kps[idx["left_wrist"]], kps[idx["right_wrist"]]
        la, ra = kps[idx["left_ankle"]], kps[idx["right_ankle"]]
        l_angle, r_angle = arm_angle(ls, lw), arm_angle(rs, rw)
        h = img.height
        feet_in = max(la[1], ra[1]) < 0.98 * h
        tpose = l_angle < 15 and r_angle < 15
        print(f"{path}: persons={n} L_arm={l_angle:.0f}deg R_arm={r_angle:.0f}deg "
              f"feet_in_frame={feet_in} strict_tpose={'PASS' if tpose else 'FAIL'} "
              f"min_kp_conf={float(np.min(conf)):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
