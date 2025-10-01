import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

class FaceModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Устройство: {self.device}")
        self.model = self._load_model()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        model = models.resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval().to(self.device)
        return model

    def extract_features(self, face_image):
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
            features = torch.flatten(features, 1).squeeze(0).cpu().numpy().astype(np.float32)
        return features
