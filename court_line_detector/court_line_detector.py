import torch
import torchvision.transforms as T
import torchvision.models as models
import cv2
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def predict(self, frame):
        h, w = frame.shape[:2]
        inp = self.transform(frame).unsqueeze(0)

        with torch.no_grad():
            kps = self.model(inp).squeeze().numpy()  # (28,)

        # scale from 224×224 back to original frame size
        kps[0::2] *= w / 224  # x coords
        kps[1::2] *= h / 224  # y coords
        return kps  # shape (28,) → 14 (x,y) pairs
