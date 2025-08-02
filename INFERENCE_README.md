# AC-MambaSeg Inference Guide

This guide explains how to use the AC-MambaSeg model for inference on new images.

## Files Overview

- `inference.py`: Main inference class and command-line interface
- `example_inference.py`: Example usage scenarios
- `INFERENCE_README.md`: This documentation file

## Quick Start

### Command Line Usage

```bash
# Basic inference
python inference.py --checkpoint path/to/checkpoint.ckpt --image path/to/image.jpg

# Save prediction mask
python inference.py --checkpoint path/to/checkpoint.ckpt --image path/to/image.jpg --output prediction.png

# Visualize results
python inference.py --checkpoint path/to/checkpoint.ckpt --image path/to/image.jpg --visualize

# Custom threshold
python inference.py --checkpoint path/to/checkpoint.ckpt --image path/to/image.jpg --threshold 0.7

# Use CPU
python inference.py --checkpoint path/to/checkpoint.ckpt --image path/to/image.jpg --device cpu
```

### Python API Usage

```python
from inference import AC_MambaSegInference

# Initialize inference
inferencer = AC_MambaSegInference("path/to/checkpoint.ckpt")

# Single image prediction
result = inferencer.predict("path/to/image.jpg", threshold=0.5)
print(f"Confidence: {result['confidence']:.4f}")

# Save prediction
inferencer.save_prediction(result, "prediction.png")

# Visualize results
inferencer.visualize_prediction("path/to/image.jpg", result)
```

## Features

### 1. Flexible Model Loading
- Supports PyTorch Lightning checkpoints
- Supports direct model weights
- Automatic device detection (CUDA/CPU)

### 2. Image Preprocessing
- Automatic resizing to model input size (192x256)
- Normalization using ImageNet statistics
- RGB conversion and tensor formatting

### 3. Prediction Options
- **Single Image**: Process one image at a time
- **Batch Processing**: Process multiple images efficiently
- **Custom Thresholds**: Adjust segmentation sensitivity
- **Confidence Scoring**: Get prediction confidence levels

### 4. Output Options
- **Binary Masks**: Clean segmentation masks
- **Probability Maps**: Raw prediction probabilities
- **Visualizations**: Side-by-side comparisons
- **Multiple Formats**: PNG, JPG, etc.

### 5. Visualization Features
- Original image display
- Prediction probability heatmap
- Binary segmentation mask
- Side-by-side comparison

## API Reference

### AC_MambaSegInference Class

#### Constructor
```python
AC_MambaSegInference(checkpoint_path, device='auto')
```

**Parameters:**
- `checkpoint_path` (str): Path to trained model checkpoint
- `device` (str): 'cuda', 'cpu', or 'auto'

#### Methods

##### predict(image_path, threshold=0.5)
Perform inference on a single image.

**Parameters:**
- `image_path` (str): Path to input image
- `threshold` (float): Segmentation threshold (0.0-1.0)

**Returns:**
- `dict`: Contains 'prediction', 'binary_mask', and 'confidence'

##### predict_batch(image_paths, threshold=0.5)
Process multiple images.

**Parameters:**
- `image_paths` (list): List of image paths
- `threshold` (float): Segmentation threshold

**Returns:**
- `list`: List of prediction results

##### save_prediction(result, save_path, format='png')
Save prediction mask to file.

**Parameters:**
- `result` (dict): Prediction result from predict()
- `save_path` (str): Output file path
- `format` (str): Image format

##### visualize_prediction(image_path, result, save_path=None)
Create visualization of results.

**Parameters:**
- `image_path` (str): Original image path
- `result` (dict): Prediction result
- `save_path` (str, optional): Path to save visualization

## Examples

### Example 1: Basic Inference
```python
from inference import AC_MambaSegInference

# Load model
inferencer = AC_MambaSegInference("model.ckpt")

# Predict
result = inferencer.predict("skin_lesion.jpg")
print(f"Confidence: {result['confidence']:.4f}")

# Save result
inferencer.save_prediction(result, "segmentation.png")
```

### Example 2: Batch Processing
```python
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = inferencer.predict_batch(image_paths)

for i, result in enumerate(results):
    inferencer.save_prediction(result, f"prediction_{i+1}.png")
```

### Example 3: Custom Thresholds
```python
thresholds = [0.3, 0.5, 0.7]
for threshold in thresholds:
    result = inferencer.predict("image.jpg", threshold=threshold)
    inferencer.save_prediction(result, f"pred_thresh_{threshold}.png")
```

### Example 4: Visualization
```python
result = inferencer.predict("image.jpg")
inferencer.visualize_prediction("image.jpg", result, "comparison.png")
```

## Requirements

### Dependencies
```bash
pip install torch torchvision
pip install pillow matplotlib
pip install mamba_ssm einops timm
```

### Model Checkpoint
You need a trained AC-MambaSeg model checkpoint. The checkpoint should contain:
- Model state dictionary
- Training configuration (optional)

## Troubleshooting

### Common Issues

1. **Checkpoint not found**
   - Verify the checkpoint path is correct
   - Ensure the file exists and is readable

2. **CUDA out of memory**
   - Use `--device cpu` for CPU inference
   - Reduce batch size if processing multiple images

3. **Image loading errors**
   - Ensure image format is supported (JPG, PNG, etc.)
   - Check file permissions and path

4. **Model loading errors**
   - Verify checkpoint is compatible with current model architecture
   - Check if all required dependencies are installed

### Performance Tips

1. **GPU Usage**: Use CUDA for faster inference
2. **Batch Processing**: Process multiple images together when possible
3. **Memory Management**: Close visualizations to free memory
4. **Threshold Tuning**: Adjust threshold based on your use case

## Advanced Usage

### Custom Preprocessing
```python
# Modify the transform in the class
inferencer.transform = transforms.Compose([
    transforms.Resize((384, 512)),  # Custom size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### Integration with Other Tools
```python
# Save as numpy array
import numpy as np
np.save("prediction.npy", result['prediction'])

# Convert to PIL Image
from PIL import Image
mask = Image.fromarray((result['binary_mask'] * 255).astype(np.uint8))
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model checkpoint is compatible
4. Check input image format and size