import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
from models.AC_MambaSeg import AC_MambaSeg

class AC_MambaSegInference:
    def __init__(self, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the inference class for AC-MambaSeg model
        
        Args:
            checkpoint_path (str): Path to the trained model checkpoint
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = AC_MambaSeg()
        
        # Load the trained model
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in checkpoint:
                # Lightning checkpoint
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Direct model weights
                self.model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        self.model.to(device)
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((192, 256)),  # Match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {device}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for inference
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        # Apply preprocessing
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, image_path, threshold=0.5):
        """
        Perform inference on a single image
        
        Args:
            image_path (str): Path to the input image
            threshold (float): Threshold for binary segmentation (default: 0.5)
            
        Returns:
            dict: Dictionary containing prediction results
        """
        with torch.no_grad():
            # Preprocess image
            input_tensor = self.preprocess_image(image_path)
            input_tensor = input_tensor.to(self.device)
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Apply sigmoid and threshold
            prediction = torch.sigmoid(output)
            binary_mask = (prediction > threshold).float()
            
            # Convert to numpy for visualization
            prediction_np = prediction.cpu().numpy()[0, 0]  # Remove batch and channel dims
            binary_mask_np = binary_mask.cpu().numpy()[0, 0]
            
            return {
                'prediction': prediction_np,
                'binary_mask': binary_mask_np,
                'confidence': prediction_np.max()
            }
    
    def predict_batch(self, image_paths, threshold=0.5):
        """
        Perform inference on multiple images
        
        Args:
            image_paths (list): List of image paths
            threshold (float): Threshold for binary segmentation
            
        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, threshold)
            results.append(result)
        return results
    
    def visualize_prediction(self, image_path, prediction_result, save_path=None):
        """
        Visualize the prediction results
        
        Args:
            image_path (str): Path to the original image
            prediction_result (dict): Result from predict() method
            save_path (str): Path to save the visualization (optional)
        """
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        original_image = original_image.resize((256, 192))  # Match model output size
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction mask
        axes[1].imshow(prediction_result['prediction'], cmap='hot')
        axes[1].set_title('Prediction Mask')
        axes[1].axis('off')
        
        # Binary mask
        axes[2].imshow(prediction_result['binary_mask'], cmap='gray')
        axes[2].set_title('Binary Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_prediction(self, prediction_result, save_path, format='png'):
        """
        Save the prediction mask to file
        
        Args:
            prediction_result (dict): Result from predict() method
            save_path (str): Path to save the prediction
            format (str): Image format ('png', 'jpg', etc.)
        """
        # Convert prediction to PIL Image
        prediction_img = Image.fromarray((prediction_result['binary_mask'] * 255).astype(np.uint8))
        prediction_img.save(save_path, format=format.upper())
        print(f"Prediction saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='AC-MambaSeg Inference')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the prediction mask')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization of results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary segmentation (default: 0.5)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize inference
    try:
        inferencer = AC_MambaSegInference(args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Perform prediction
    try:
        result = inferencer.predict(args.image, args.threshold)
        print(f"Prediction completed with confidence: {result['confidence']:.4f}")
        
        # Save prediction if output path is provided
        if args.output:
            inferencer.save_prediction(result, args.output)
        
        # Visualize if requested
        if args.visualize:
            inferencer.visualize_prediction(args.image, result)
            
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()