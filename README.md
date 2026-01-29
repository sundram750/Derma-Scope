# 🔬 Dermo-Scope: Real-Time Skin Disease Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io/)

A real-time skin disease detection system using deep learning and augmented reality. Built with MobileNetV2, TensorFlow, and Streamlit, featuring Grad-CAM explainability for high-risk predictions.

## 🎯 Features

- **Real-time Detection**: Live webcam analysis for skin disease classification
- **7 Disease Classes**: Detects Melanoma, BCC, Actinic Keratoses, Nevi, and more
- **Grad-CAM Visualization**: Explainable AI showing model attention areas
- **Risk Assessment**: Color-coded risk levels (High/Monitor/Low)
- **Interactive UI**: Modern Streamlit web interface
- **MobileNetV2 Architecture**: Fast and accurate predictions

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (for AR web application)
- HAM10000 dataset (download separately)
- 4GB+ RAM recommended
- (Optional) NVIDIA GPU for faster training

## 🚀 Quick Start

### 1. Installation

Clone or download this project, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

1. Download the HAM10000 dataset (or ensure you have `archive.zip`)
2. Extract `archive.zip` to the `raw_data/` folder:
   ```
   raw_data/
   ├── HAM10000_metadata.csv
   ├── ISIC_0024306.jpg
   ├── ISIC_0024307.jpg
   └── ... (more images)
   ```

### 3. Organize Dataset

Run the data organization script:

```bash
python data_tools/01_organize_data.py
```

This will organize images into 7 disease-specific folders in `organized_data/`.

### 4. Train the Model

Train the MobileNetV2-based classifier:

```bash
python model_training/02_train_model.py
```

**Note**: Training may take 30-60 minutes depending on your hardware. The script will:
- Use data augmentation
- Train for up to 20 epochs (with early stopping)
- Save the best model as `model_training/skin_model.h5`

### 5. Launch the AR Web App

Start the Streamlit application:

```bash
streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
major project/
├── data_tools/
│   └── 01_organize_data.py       # Dataset organization script
├── model_training/
│   ├── 02_train_model.py         # Model training script
│   ├── skin_model.h5             # Trained model (after training)
│   └── training_history.png      # Training plots
├── app/
│   └── main.py                   # Streamlit AR web application
├── raw_data/                     # Raw dataset (you provide)
│   ├── HAM10000_metadata.csv
│   └── [image files]
├── organized_data/               # Organized by disease class
│   ├── nv/                      # Melanocytic nevi
│   ├── mel/                     # Melanoma
│   ├── bcc/                     # Basal cell carcinoma
│   ├── akiec/                   # Actinic keratoses
│   ├── bkl/                     # Benign keratosis
│   ├── df/                      # Dermatofibroma
│   └── vasc/                    # Vascular lesions
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🎨 Disease Classes

| Code  | Disease Name                    | Risk Level | Description                                      |
|-------|---------------------------------|------------|--------------------------------------------------|
| mel   | Melanoma                        | 🔴 HIGH    | Serious skin cancer, requires immediate attention|
| bcc   | Basal Cell Carcinoma            | 🔴 HIGH    | Most common skin cancer                          |
| akiec | Actinic Keratoses               | 🟡 MONITOR | Precancerous patches from sun damage             |
| nv    | Melanocytic Nevi                | 🟢 LOW     | Common benign moles                              |
| bkl   | Benign Keratosis                | 🟢 LOW     | Non-cancerous skin growths                       |
| df    | Dermatofibroma                  | 🟢 LOW     | Benign fibrous nodules                           |
| vasc  | Vascular Lesions                | 🟢 LOW     | Blood vessel abnormalities                       |

## 🧠 Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**: GlobalAveragePooling2D + Dense(128) + Dense(7)
- **Input Size**: 224×224×3 RGB images
- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Categorical Crossentropy
- **Data Augmentation**: Rotation, flip, zoom, shift

## 🔍 Grad-CAM Explainability

For **HIGH RISK** predictions (Melanoma, BCC), the system automatically generates a Grad-CAM heatmap overlay:
- **Red areas**: Regions the AI focused on for classification
- **Transparency**: Blended at 40% opacity
- **Purpose**: Provides transparency and builds trust in AI decisions

## 🛠️ Troubleshooting

### Issue: "Model not found"
**Solution**: Run the training script first: `python model_training/02_train_model.py`

### Issue: "Metadata file not found"
**Solution**: Ensure `HAM10000_metadata.csv` is in the `raw_data/` folder

### Issue: Webcam not working
**Solution**: 
- Check browser permissions for webcam access
- Use Chrome or Firefox (WebRTC support required)
- For production deployment, HTTPS is required

### Issue: Low training accuracy
**Solution**: 
- Ensure sufficient dataset (recommended: 1000+ images)
- Try training for more epochs
- Check data quality and class balance

### Issue: Slow predictions
**Solution**: 
- Use a GPU if available (TensorFlow will auto-detect)
- Reduce video frame rate in `app/main.py`
- Consider model quantization for edge deployment

## ⚠️ Disclaimer

**This is a prototype AI tool for educational and research purposes only.**

- NOT a substitute for professional medical diagnosis
- NOT clinically validated
- Always consult qualified dermatologists for skin concerns
- The model's predictions should be interpreted by medical professionals

## 📊 Performance Notes

- Expected validation accuracy: 70-85% (depends on dataset size)
- Inference time: ~50-100ms per frame (CPU), ~10-20ms (GPU)
- Model size: ~15MB (MobileNetV2 is lightweight)

## 🔧 Advanced Usage

### Custom Training Configuration

Edit training parameters in `model_training/02_train_model.py`:

```python
EPOCHS = 20              # Number of training epochs
BATCH_SIZE = 32          # Batch size
LEARNING_RATE = 0.0001   # Learning rate
```

### Fine-tuning the Base Model

To unfreeze MobileNetV2 layers for fine-tuning:

```python
# In model_training/02_train_model.py
base_model.trainable = True  # Unfreeze
```

### Customizing the UI

Modify `app/main.py` to change:
- Color schemes
- Risk thresholds
- Heatmap opacity
- UI layout

## 📚 Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Image processing and visualization
- **Streamlit**: Web application framework
- **Streamlit-WebRTC**: Real-time video streaming
- **MobileNetV2**: Efficient CNN architecture
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization

## 📖 Dataset Reference

HAM10000 Dataset:
- **Title**: "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions"
- **Authors**: Tschandl, P., Rosendahl, C. & Kittler, H.
- **Published**: Scientific Data (2018)

## 🤝 Contributing

This is a prototype project. For improvements:
1. Increase dataset size for better accuracy
2. Implement data balancing techniques
3. Add more disease classes
4. Deploy to cloud with HTTPS support
5. Add user authentication
6. Integrate with medical record systems (HIPAA compliant)

## 📄 License

This project is for educational purposes. Please respect the HAM10000 dataset licensing terms.

## 👨‍💻 Author

Dermo-Scope Team

---

**Built with ❤️ for healthcare AI research**
#Derma_Scope
