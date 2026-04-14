import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import cv2
import numpy as np


class CourtLineDetector:
    """
    ResNet-50 based regression model to detect 14 court keypoints.
    Input : single RGB frame
    Output: 28 values → 14 (x, y) pairs representing court landmarks
    """

    def __init__(self, model_path):
        # build ResNet-50 with custom output head
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 14 * 2)

        # load trained weights
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        self.model.eval()

        # ImageNet normalization (ResNet expects this)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, frame):
        """
        Predict 14 court keypoints from a single frame.
        Returns numpy array of shape (28,) → [x0,y0, x1,y1, ... x13,y13]
        Coordinates are scaled back to original frame resolution.
        """
        h, w = frame.shape[:2]

        # preprocess
        inp = self.transform(frame).unsqueeze(0)   # (1, 3, 224, 224)

        with torch.no_grad():
            kps = self.model(inp).squeeze().numpy()  # (28,)

        # scale from 224x224 back to original frame size
        kps[0::2] *= (w / 224)   # x coordinates
        kps[1::2] *= (h / 224)   # y coordinates

        return kps   # shape (28,)
