#!/usr/bin/env python3
"""
Example script demonstrating how to use AC-MambaSeg for inference
"""

import os
from inference import AC_MambaSegInference

def example_single_image():
    """Example: Inference on a single image"""
    print("=== Single Image Inference Example ===")
    
    # Initialize the inference class
    checkpoint_path = "path/to/your/checkpoint.ckpt"  # Update this path
    inferencer = AC_MambaSegInference(checkpoint_path)
    
    # Perform prediction
    image_path = "path/to/your/image.jpg"  # Update this path
    result = inferencer.predict(image_path, threshold=0.5)
    
    print(f"Prediction confidence: {result['confidence']:.4f}")
    
    # Save the prediction
    inferencer.save_prediction(result, "prediction_mask.png")
    
    # Visualize the results
    inferencer.visualize_prediction(image_path, result, "visualization.png")

def example_batch_inference():
    """Example: Inference on multiple images"""
    print("=== Batch Inference Example ===")
    
    # Initialize the inference class
    checkpoint_path = "path/to/your/checkpoint.ckpt"  # Update this path
    inferencer = AC_MambaSegInference(checkpoint_path)
    
    # List of images to process
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg"
    ]
    
    # Perform batch prediction
    results = inferencer.predict_batch(image_paths, threshold=0.5)
    
    # Process results
    for i, (image_path, result) in enumerate(zip(image_paths, results)):
        print(f"Image {i+1}: {os.path.basename(image_path)}")
        print(f"  Confidence: {result['confidence']:.4f}")
        
        # Save individual predictions
        output_path = f"prediction_{i+1}.png"
        inferencer.save_prediction(result, output_path)

def example_custom_threshold():
    """Example: Using different thresholds for segmentation"""
    print("=== Custom Threshold Example ===")
    
    # Initialize the inference class
    checkpoint_path = "path/to/your/checkpoint.ckpt"  # Update this path
    inferencer = AC_MambaSegInference(checkpoint_path)
    
    image_path = "path/to/your/image.jpg"  # Update this path
    
    # Try different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        result = inferencer.predict(image_path, threshold=threshold)
        print(f"Threshold {threshold}: Confidence = {result['confidence']:.4f}")
        
        # Save with threshold in filename
        output_path = f"prediction_threshold_{threshold}.png"
        inferencer.save_prediction(result, output_path)

def example_cpu_inference():
    """Example: Running inference on CPU"""
    print("=== CPU Inference Example ===")
    
    # Initialize the inference class with CPU
    checkpoint_path = "path/to/your/checkpoint.ckpt"  # Update this path
    inferencer = AC_MambaSegInference(checkpoint_path, device='cpu')
    
    image_path = "path/to/your/image.jpg"  # Update this path
    result = inferencer.predict(image_path)
    
    print(f"CPU inference completed with confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    print("AC-MambaSeg Inference Examples")
    print("=" * 40)
    
    # Uncomment the example you want to run:
    
    # example_single_image()
    # example_batch_inference()
    # example_custom_threshold()
    # example_cpu_inference()
    
    print("\nTo run examples:")
    print("1. Update the checkpoint and image paths in the functions")
    print("2. Uncomment the function call you want to test")
    print("3. Run: python example_inference.py")