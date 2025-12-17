# Smart Farm Leaf Detection and Disease Identification System

**Capstone Project (Fall 2024)**: Image Processing and AI Technology for Crop Quality Assessment

This project develops a lightweight object detection system for smart farm environments to accurately count crop leaves and detect abnormal leaves (disease and pest damage). The system enhances automation and precision in agricultural management, enabling real-time crop health monitoring and rapid disease response.

## 🎯 Project Overview

### Motivation

- The expanding smart farm market demands advanced disease recognition and intervention capabilities during crop growth cycles
- Plant diseases significantly impact crop yield, quality, and profitability
- Traditional manual detection methods are time-consuming and inefficient, necessitating automated detection technologies
- Real-time monitoring systems are essential for precision agriculture and sustainable farming practices

### Key Capabilities

- **Quantitative Leaf Counting**: Automated leaf enumeration through advanced image analysis
- **Abnormal Leaf Detection**: Identification of disease-affected and pest-damaged foliage
- **Real-time Health Status Assessment**: Classification of crop conditions ("Normal", "Caution", "Warning", "Danger", "Critical")
- **Growth Prediction Framework**: Foundation for developing predictive models based on collected temporal data

## 🔬 Technical Approach

### Model Architecture & Development

**Core Model**: YOLOv8 (You Only Look Once v8)
- Selected for optimal balance between detection accuracy and computational efficiency
- Lightweight architecture suitable for edge deployment in smart farm environments
- Real-time inference capability for continuous monitoring systems

**Model Development Process**:

1. **Data Collection & Curation**
   - Multi-source dataset compilation:
     - Industry-provided proprietary data
     - Kaggle open-source datasets
     - Real-world field data collection
   - Comprehensive data augmentation pipeline achieving 60× dataset expansion
   - Strategic augmentation techniques: rotation, scaling, color jittering, and synthetic occlusion

2. **Annotation & Preprocessing**
   - Precise object labeling using RoboFlow platform
   - Multi-class annotation: `healthy_leaf` and `diseased_leaf`
   - Quality control measures ensuring annotation consistency
   - Train/validation/test split optimization (70/20/10)

3. **Model Training & Optimization**
   - Transfer learning from COCO pre-trained weights
   - Hyperparameter tuning for optimal performance
   - Class imbalance handling through weighted loss functions
   - Early stopping and learning rate scheduling for convergence optimization

4. **Performance Evaluation**
   - Comprehensive metric analysis:
     - Mean Average Precision (mAP@0.5): **0.810**
     - Diseased leaf class AP: **0.846**
     - Healthy leaf class AP: **0.774**
   - Precision-Recall curve analysis
   - Confusion matrix evaluation for error analysis

### Computer Vision Pipeline

**Detection Algorithm**:
```
Input Image → Preprocessing → YOLO Inference → Post-processing → Status Assessment
```

1. **Preprocessing**: Image normalization and resizing for model input
2. **Object Detection**: YOLOv8 inference generating bounding boxes and class predictions
3. **Post-processing**: Non-maximum suppression (NMS) for overlapping detection removal
4. **Health Ratio Calculation**: 
```
   Health_Ratio = (Healthy_Leaves / Total_Leaves) × 100%
```
5. **Status Classification**:
   - **Normal** (≥80% healthy): Green indicator
   - **Caution** (60-80%): Yellow indicator
   - **Warning** (40-60%): Orange indicator
   - **Danger** (20-40%): Red indicator
   - **Critical** (<20%): Purple indicator

### Visual Output Generation

- **Bounding Box Visualization**: Class-specific color coding for detected objects
- **Status Overlay**: Real-time health assessment displayed on image
- **Confidence Scores**: Detection confidence for each identified leaf
- **Metadata Annotation**: Leaf counts and health metrics

## 🚀 Installation & Usage

### Prerequisites

Virtual environment setup recommended (Anaconda or virtualenv)

### Setup Instructions

1. **Clone Repository**
```bash
git clone https://github.com/andimsewon/Capstone_GuardiansofFarm
cd Capstone_GuardiansofFarm
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Running Detection

1. **Prepare Input Images**
   - Place images in the `chocomint` folder

2. **Execute Detection**
```bash
python detector.py
```
   - Output images automatically saved to `chocomint_result` folder
   - Results include annotated bounding boxes and health status assessment

### Output Interpretation

Each processed image contains:
- Total leaf count (healthy + diseased)
- Disease detection results with confidence scores
- Overall plant health status classification
- Visual indicators for immediate assessment

## 📊 Performance Metrics & Results

### Model Performance

| Metric | Score |
|--------|-------|
| mAP@0.5 (Overall) | 0.810 |
| Diseased Leaf AP | 0.846 |
| Healthy Leaf AP | 0.774 |
| Inference Time | <50ms per image |

### Key Achievements

- **High Precision Disease Detection**: 84.6% AP for diseased leaf classification enables reliable early disease identification
- **Real-time Processing**: Lightweight model architecture supports continuous monitoring applications
- **Robust Generalization**: Multi-source training data ensures performance across diverse crop conditions
- **Scalable Framework**: Modular design facilitates extension to additional crop types and disease categories

## 🔮 Future Directions & Research Opportunities

### Immediate Enhancements

1. **Dataset Expansion**
   - Incorporate additional disease phenotypes
   - Multi-crop species support
   - Temporal progression datasets for disease development modeling

2. **Model Improvements**
   - Multi-scale detection for varying leaf sizes
   - Attention mechanisms for fine-grained disease classification
   - Ensemble methods combining multiple detection architectures

3. **System Integration**
   - IoT sensor fusion (temperature, humidity, soil conditions)
   - Edge deployment optimization for resource-constrained environments
   - Cloud-based aggregation for farm-wide analytics

### Vision-Language Model (VLM) Integration

This project establishes a foundation for advanced VLM research directions:

- **Natural Language Disease Descriptions**: Generating interpretable diagnostic reports from visual observations
- **Multimodal Query Systems**: Enabling farmers to query crop status using natural language
- **Knowledge-Grounded Recommendations**: Integrating agricultural domain knowledge with visual understanding
- **Few-shot Learning**: Adapting to novel diseases with minimal labeled examples

The intersection of computer vision and language understanding in agricultural contexts presents compelling research opportunities, particularly in developing accessible AI systems for precision farming applications.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork this repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## 📧 Contact

**Sewon Kim**
- Email: sewonkim1018@gmail.com
- GitHub: [github.com/andimsewon](https://github.com/andimsewon)
- LinkedIn: [linkedin.com/in/sewon-kim-742a492a6](https://www.linkedin.com/in/sewon-kim-742a492a6/)

---

## 📝 Project Significance

This project demonstrates expertise in:
- **Computer Vision Systems**: End-to-end development from data collection to deployment
- **Deep Learning Architecture**: Model selection, training, and optimization
- **Agricultural AI Applications**: Domain-specific problem-solving in precision agriculture
- **Research Methodology**: Systematic approach to technical challenges with quantitative evaluation

The work represents a practical application of advanced AI techniques to real-world problems, showcasing both technical proficiency and domain awareness essential for impactful research in vision-language models and multimodal AI systems.

---

*Developed as part of Industry-Academic Capstone Project, Fall 2024*

**Built with passion for advancing AI in agriculture 🌱**
