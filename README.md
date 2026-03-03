# 🌿 Terrain Segregation - Off-Road Semantic Segmentation

**Team Name:** !perfect  
**Project Name:** Terrain Segregation  
**Event:** Hack For Green Bharat  

This repository contains the machine learning training and evaluation pipeline for our submission to the **Hack For Green Bharat** hackathon. The project focuses on multi-class off-road semantic segmentation using a robust deep learning architecture to accurately classify various natural and terrain elements.

## 🏆 Key Results
* **Best Validation mIoU: 0.8290** (Achieved at Epoch 45)

## 🔬 Model Selection & Ablation Study
To achieve the highest possible Intersection over Union (IoU) on complex, unstructured terrain, we rigorously tested multiple state-of-the-art architectures and backbones. Our experiments included **SegFormer, Mask2Former, MobileNet, and various ImageNet-pretrained CNNs (including other EfficientNet variants)**. 

Ultimately, the combination of **EfficientNet-B5 (Encoder)** and **U-Net++ (Decoder)** significantly outperformed the rest. Here is a brief breakdown of our findings and why the other models fell short:

* **SegFormer:** While vision transformers are excellent at capturing global context, they often struggle to smoothly delineate highly irregular, chaotic spatial boundaries (like overlapping dry grass vs. dense bushes) without massive amounts of data. The dense, nested skip connections in U-Net++ proved vastly superior at recovering these fine-grained local textures. 
* **Mask2Former:** Despite being a highly powerful universal architecture, it is notoriously data-hungry. The complex bipartite matching and query-based learning were prone to overfitting on our specific off-road dataset size, struggling to converge as cleanly as feature-pyramid CNN approaches.
* **MobileNet:** While lightweight and fast for mobile deployment, it lacked the representational capacity (depth and parameters) to effectively differentiate between visually similar, complex terrain classes (e.g., distinguishing "Lush Bushes" from "Trees" or "Dry Grass").
* **Other EfficientNet Variants:** Lower-capacity variants (B0-B4) underfit the complex scenes, lacking the compound scaling needed for rich texture extraction. B5 hit the perfect "sweet spot" of receptive field and parameter capacity without causing the network to memorize the training set.

## 🧠 Winning Architecture
The final model leverages the `segmentation-models-pytorch` (SMP) library:
* **Encoder:** `EfficientNet-B5` (Pre-trained on ImageNet for robust baseline feature extraction)
* **Decoder:** `U-Net++` (For highly accurate, fine-grained spatial reconstruction)
* **Classes:** 10

## 📊 Dataset & Classes
The dataset consists of high-resolution off-road color images and their corresponding segmentation masks. The 10 predicted classes are:
0. `Trees` (100)
1. `Lush Bushes` (200)
2. `Dry Grass` (300)
3. `Dry Bushes` (500)
4. `Ground Clutter` (550)
5. `Flowers` (600)
6. `Logs` (700)
7. `Rocks` (800)
8. `Landscape` (7100)
9. `Sky` (10000)

## 🛠️ Training Pipeline

### 1. Data Augmentation (`Albumentations`)
To prevent overfitting and make the model robust to various off-road lighting and weather conditions, a heavy augmentation pipeline was utilized:
* Spatial: Horizontal/Vertical Flips, ShiftScaleRotate, Perspective, Optical/Grid Distortions.
* Pixel-level: GaussNoise, RandomBrightnessContrast, RandomGamma, HueSaturationValue.
* Dropout: CoarseDropout (Cutout).
* Image Resizing: 512x512.

### 2. Loss Function
A custom `CombinedLoss` function was used to tackle class imbalances and ensure sharp boundary predictions:
* **Jaccard Loss (IoU)** + **Focal Loss**

### 3. Optimizer & Scheduler
* **Optimizer:** AdamW (`LR = 3e-4`, `Weight Decay = 1e-4`)
* **Learning Rate Scheduler:** OneCycleLR
* **Fine-Tuning:** The model was carefully fine-tuned from epoch 21 to 50 with a lowered learning rate (`5e-5`) to push the mIoU past the 0.82+ mark.

### 4. Hardware
* Training utilizes PyTorch Automatic Mixed Precision (AMP) (`torch.amp.autocast`) for faster training and reduced memory consumption, optimized for `NVIDIA H100` / `T4` accelerators.

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install segmentation-models-pytorch albumentations==1.3.1 opencv-python pandas matplotlib
