# **Synthetic Image Generation, Preprocessing, and Flux Model Forward Pass**  
![Alt text](https://github.com/gaganchapa/Padel_Start/blob/main/diagram-export-07-03-2025-23_14_31.png)

## **Overview**  
This project involves generating synthetic images using a diffusion model, preprocessing images, defining a Convolutional Neural Network (CNN) using Flux.jl, and extracting feature maps to visualize model activations.  

## **Workflow Breakdown**  
1. **Synthetic Image Generation** - Generate images using a text-to-image diffusion model.  
2. **Image Preprocessing & Model Processing** - Prepare images for deep learning and perform a forward pass through a CNN.  
3. **Feature Extraction & Visualization** - Extract and visualize intermediate feature maps from the model.  

---  

## **1. Synthetic Image Generation (`diffusion.ipynb`)**  
- Uses a **Stable Diffusion** model to generate images from a text prompt (e.g., *"a serene sunset over a futuristic city"*).  
- Generates at least **three** images based on the prompt.  
- Saves the images to disk with appropriate filenames.  
- If local generation is not feasible due to hardware constraints, pre-generated images are used.  

### **Running Synthetic Image Generation**  
```sh  
jupyter notebook diffusion.ipynb  
```

---  

## **2. Image Preprocessing & Model Processing (`code.jl`)**  
This step processes the generated images and runs them through a CNN in Flux.jl.  

### **Steps Involved:**  
1. **Image Preprocessing**  
   - Loads the generated images and verifies their existence.  
   - Resizes images to **224×224 pixels**.  
   - Converts images to grayscale if necessary.  
   - Transforms images into **Float32 tensors** for model input.  
   - If image processing fails, generates a fallback test pattern.  
   - Saves the preprocessed image as `loaded_image_verification.png`.  

2. **CNN Model Definition**  
   - Implements a **simple CNN** with three convolutional layers.  
   - Uses **ReLU activation** and **Max-pooling** for feature extraction.  
   - Includes **two dense layers** and a final **softmax activation** for classification.  
   
### **Model Architecture in Flux**  
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

### **Running Image Preprocessing & Model Processing**  
```sh  
# Install dependencies for Julia  
julia req.jl  

# Run the main script for processing and feature extraction  
julia code.jl  
```

---  

## **3. Feature Extraction & Visualization**  
- Performs a **forward pass** using the preprocessed image.  
- Extracts **intermediate feature maps** from CNN layers.  
- Normalizes and saves feature maps as heatmaps.  
- Generates additional visualizations such as contour and surface plots.  
- Saves outputs to disk for interpretation.  

### **Output Files**  
| File | Description |  
|------|-------------|  
| `loaded_image_verification.png` | Preprocessed image for verification |  
| `conv_layer_X_feature_maps.png` | Extracted feature maps from convolutional layers |  
| `model_architecture.txt` | Saved model structure |  
| `input_image_visualizations.png` | Contour and surface plots of input image |  

---  

## **Final Notes**  
- Ensure that all dependencies are installed before running scripts.  
- If GPU is available, the model will utilize CUDA for faster computation.  
- Outputs are saved automatically in the working directory.  


