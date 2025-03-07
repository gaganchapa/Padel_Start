# Synthetic Image Generation, Preprocessing, and Flux Model Forward Pass


## Overview
This project involves processing images, defining a Convolutional Neural Network (CNN) using the Flux.jl library, and visualizing feature maps. The workflow consists of:

1. **Image Preprocessing** - Loading, resizing, and converting images to a tensor format suitable for deep learning.
2. **CNN Model Definition** - A simple CNN is implemented using Flux.jl to perform a forward pass.
3. **Feature Map Extraction & Visualization** - Extracting intermediate feature maps from convolutional layers and saving visualizations.

## Workflow
### 1. Image Preprocessing
- The script loads an input image and checks for its existence.
- The image is resized to **224×224 pixels** and converted to grayscale if necessary.
- The preprocessed image is transformed into a **Float32 tensor** and reshaped for CNN input.
- If the image processing fails, a test pattern is generated instead.
- The processed image is saved as a heatmap for verification.

### 2. CNN Model Definition
A simple CNN model is created using Flux.jl. It consists of:

- **Three convolutional layers** with ReLU activation.
- **Max-pooling layers** to reduce spatial dimensions.
- A **fully connected layer** followed by a **softmax activation** for classification.

#### Model Architecture in Flux
```julia
using Flux

model = Chain(
    Conv((3, 3), 1 => 16, relu),
    MaxPool((2,2)),
    Conv((3, 3), 16 => 32, relu),
    MaxPool((2,2)),
    Conv((3, 3), 32 => 64, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(43264, 128, relu),
    Dense(128, 10),
    softmax
)
```

### 3. Feature Map Extraction & Visualization
- The forward pass is performed using the preprocessed image.
- Intermediate feature maps are extracted from convolutional layers.
- These feature maps are normalized and plotted as heatmaps.
- The feature maps are saved as images for interpretation.

## Running the Project
### 1. Install Required Packages
Ensure you have all necessary Julia packages installed by running:
```julia
# Install dependencies for Julia
julia req.jl
```

### 2. Run the Code
Execute the scripts in order:
```sh
# Generate synthetic images using diffusion model
jupyter notebook diffusion.ipynb

# Run the main script for processing and feature extraction
julia code.jl
```

## Outputs
- **Preprocessed Image:** Saved as `loaded_image_verification.png`
- **Feature Maps:** Extracted and saved as `conv_layer_X_feature_maps.png`
- **Model Architecture:** Saved as `model_architecture.txt`
- **Additional Visualizations:** Contour and surface plots saved in `input_image_visualizations.png`,`input_image_visualizations/png`


