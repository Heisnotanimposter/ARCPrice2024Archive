import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

# Initialize Faster R-CNN model with updated weights parameter
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)



class FasterRCNNModel:
    def __init__(self, device='cpu', model_type='resnet50_fpn'):
        """
        Initialize the Faster R-CNN model.
        
        Args:
        - device (str): Device to load the model on ('cpu' or 'cuda').
        - model_type (str): Type of Faster R-CNN model to use ('resnet50_fpn', 'resnet50_fpn_v2', etc.).
        """
        self.device = device
        self.model = self._load_model(model_type).to(device)
        self.model.eval()

    def _load_model(self, model_type):
        """
        Load a Faster R-CNN model based on the specified type.

        Args:
        - model_type (str): Model variant to load.
        
        Returns:
        - model: Loaded Faster R-CNN model.
        """
        if model_type == 'resnet50_fpn':
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        elif model_type == 'resnet50_fpn_v2':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
        elif model_type == 'mobilenet_v3_large_fpn':
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        elif model_type == 'mobilenet_v3_large_320_fpn':
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model

    def preprocess(self, image):
        """
        Preprocess the input image for the Faster R-CNN model.

        Args:
        - image (np.ndarray): Input image as a numpy array.
        
        Returns:
        - tensor: Preprocessed image as a tensor.
        """
        image_tensor = F.to_tensor(image).unsqueeze(0).float()
        return image_tensor.to(self.device)

    def predict(self, image):
        """
        Make predictions using the Faster R-CNN model.

        Args:
        - image (np.ndarray): Input image as a numpy array.
        
        Returns:
        - predictions: Model predictions (bounding boxes, labels, scores).
        """
        with torch.no_grad():
            image_tensor = self.preprocess(image)
            predictions = self.model(image_tensor)
        return predictions

    def postprocess(self, predictions, score_threshold=0.5):
        """
        Post-process model predictions to filter based on score threshold.

        Args:
        - predictions: Raw model predictions.
        - score_threshold (float): Score threshold for filtering predictions.
        
        Returns:
        - filtered_boxes: Filtered bounding boxes.
        - filtered_scores: Scores corresponding to the filtered bounding boxes.
        - filtered_labels: Labels corresponding to the filtered bounding boxes.
        """
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for pred in predictions:
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score >= score_threshold:
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
                    filtered_labels.append(label)

        return filtered_boxes, filtered_scores, filtered_labels

