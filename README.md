# Smart Farm Leaf Detection and Disease Identification System

**Industry-Academic Capstone Project (Fall 2024)**: Advanced Image Processing and AI Technology for Automated Crop Quality Assessment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Technical Architecture](#-technical-architecture)
- [Model Development](#-model-development)
- [Installation & Usage](#-installation--usage)
- [Performance Analysis](#-performance-analysis)
- [Research Contributions](#-research-contributions)
- [Future Directions](#-future-directions)
- [Contact](#-contact)

---

## 🎯 Project Overview

### Problem Statement

Modern agriculture faces critical challenges in crop disease management:
- **Scale**: Manual inspection is impractical for large-scale smart farms
- **Timeliness**: Early disease detection is crucial but labor-intensive
- **Expertise**: Shortage of trained personnel for accurate disease identification
- **Economic Impact**: Plant diseases cause 20-40% global crop yield losses annually

### Solution

This project develops an **AI-powered automated monitoring system** that:
- Performs real-time leaf counting and health assessment
- Detects disease symptoms with high precision (84.6% AP)
- Provides actionable health status classifications
- Enables data-driven agricultural decision-making

### Key Innovations

1. **Lightweight Architecture**: Optimized YOLOv8 model achieving <50ms inference time
2. **Multi-Source Data Integration**: Robust training on diverse agricultural datasets
3. **Interpretable Outputs**: Five-tier health classification system for intuitive monitoring
4. **Scalable Framework**: Modular design supporting extension to multiple crop species

---

## 🏗️ Technical Architecture

### System Pipeline
```
┌─────────────────┐
│  Image Capture  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  • Resize       │
│  • Normalize    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ YOLOv8 Detector │
│  • Backbone     │
│  • Neck (PANet) │
│  • Head         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-processing │
│  • NMS          │
│  • Filtering    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Health Analysis │
│  • Leaf Count   │
│  • Ratio Calc   │
│  • Status Class │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization & │
│ Output Storage  │
└─────────────────┘
```

### Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Detection Model** | YOLOv8 | State-of-the-art real-time object detection |
| **Framework** | PyTorch | Flexible deep learning ecosystem |
| **Data Augmentation** | Albumentations | Advanced image transformation pipeline |
| **Annotation Platform** | RoboFlow | Efficient collaborative labeling |
| **Visualization** | OpenCV | High-performance computer vision library |

---

## 🔬 Model Development

### 1. Dataset Construction

#### Data Sources

**Multi-Modal Data Acquisition Strategy:**
- **Industry Partnership Data**: High-quality annotated samples from commercial smart farms
- **Public Datasets**: Kaggle plant disease datasets for baseline diversity
- **Field Collection**: Custom-captured images under varied lighting and growth conditions
- **Synthetic Augmentation**: 60× expansion through systematic transformations

#### Data Augmentation Pipeline

Implemented comprehensive augmentation to enhance model robustness:
```python
Augmentation Techniques:
├── Geometric Transformations
│   ├── Random Rotation (±25°)
│   ├── Horizontal/Vertical Flipping
│   ├── Scale Variation (0.8-1.2×)
│   └── Perspective Warping
├── Color Space Adjustments
│   ├── Brightness (±15%)
│   ├── Contrast (±15%)
│   ├── Hue Shift (±10°)
│   └── Saturation (±20%)
├── Noise Injection
│   ├── Gaussian Noise (σ=0.01)
│   ├── Motion Blur
│   └── JPEG Compression Artifacts
└── Occlusion Simulation
    ├── Random Erasing
    └── Cutout
```

**Dataset Statistics:**
- **Total Images**: 12,000+ (after augmentation)
- **Annotated Leaves**: 85,000+
- **Disease Categories**: 2 classes (healthy, diseased)
- **Split Ratio**: 70% train / 20% validation / 10% test

### 2. Model Architecture & Training

#### YOLOv8 Architecture Selection

**Why YOLOv8?**
- **Speed-Accuracy Tradeoff**: Optimal balance for real-time agricultural monitoring
- **Anchor-Free Design**: Better generalization to varied leaf shapes and sizes
- **Multi-Scale Feature Fusion**: PANet architecture captures both fine details and contextual information
- **Efficient Backbone**: CSPDarknet53 provides rich feature extraction with minimal parameters

#### Training Configuration
```yaml
Model Specifications:
  Base Model: YOLOv8n (nano variant)
  Input Resolution: 640×640
  Parameters: 3.2M
  FLOPs: 8.7B

Training Hyperparameters:
  Optimizer: AdamW
  Initial Learning Rate: 1e-3
  LR Scheduler: Cosine Annealing
  Batch Size: 32
  Epochs: 300
  Weight Decay: 5e-4
  Warmup Epochs: 3

Loss Function:
  Classification: Binary Cross-Entropy
  Localization: CIoU (Complete IoU)
  Objectness: BCE with Logits
  Total Loss: λ₁·L_cls + λ₂·L_box + λ₃·L_obj
```

#### Training Strategies

**Transfer Learning Approach:**
1. **Initialization**: COCO pre-trained weights for feature extraction layers
2. **Fine-tuning**: Progressive unfreezing of backbone layers
3. **Head Specialization**: Custom detection head trained from scratch for leaf-specific features

**Class Imbalance Handling:**
- Weighted loss functions (diseased:healthy = 1.3:1.0)
- Focal loss integration for hard example mining
- Stratified sampling ensuring balanced mini-batches

**Regularization Techniques:**
- Dropout (p=0.2) in classification layers
- Label smoothing (ε=0.1)
- Mosaic augmentation during training
- MixUp for improved generalization

### 3. Computer Vision Pipeline

#### Detection Algorithm

**Inference Process:**
```python
def detect_and_classify(image):
    # Step 1: Preprocessing
    img_tensor = preprocess(image)  # Resize to 640×640, normalize
    
    # Step 2: Model Inference
    predictions = yolo_model(img_tensor)  # Forward pass
    
    # Step 3: Post-processing
    detections = apply_nms(predictions, 
                          conf_threshold=0.25,
                          iou_threshold=0.45)
    
    # Step 4: Classification & Counting
    healthy_count = sum(d.class_id == 0 for d in detections)
    diseased_count = sum(d.class_id == 1 for d in detections)
    
    # Step 5: Health Ratio Calculation
    total_leaves = healthy_count + diseased_count
    health_ratio = (healthy_count / total_leaves) * 100 if total_leaves > 0 else 0
    
    # Step 6: Status Assessment
    status = classify_health_status(health_ratio)
    
    return detections, status, health_ratio
```

#### Health Status Classification

**Five-Tier Assessment System:**

| Status | Health Ratio | Indicator | Recommended Action |
|--------|-------------|-----------|-------------------|
| 🟢 **Normal** | ≥80% | Green | Continue standard care |
| 🟡 **Caution** | 60-79% | Yellow | Increase monitoring frequency |
| 🟠 **Warning** | 40-59% | Orange | Investigate potential diseases |
| 🔴 **Danger** | 20-39% | Red | Immediate intervention required |
| 🟣 **Critical** | <20% | Purple | Emergency treatment protocol |

**Status Determination Logic:**
```python
def classify_health_status(health_ratio):
    if health_ratio >= 80:
        return Status.NORMAL
    elif health_ratio >= 60:
        return Status.CAUTION
    elif health_ratio >= 40:
        return Status.WARNING
    elif health_ratio >= 20:
        return Status.DANGER
    else:
        return Status.CRITICAL
```

### 4. Visualization & Output

**Generated Outputs:**
- **Annotated Images**: Bounding boxes with class labels and confidence scores
- **Health Dashboard**: Visual status indicator with leaf statistics
- **Detection Metadata**: JSON file containing all detection coordinates and classifications
- **Time-Series Data**: CSV logs for temporal health tracking

---

## 🚀 Installation & Usage

### Prerequisites
```bash
Python >= 3.8
CUDA >= 11.0 (optional, for GPU acceleration)
```

**Recommended**: Use Anaconda for environment management

### Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/andimsewon/Capstone_GuardiansofFarm.git
cd Capstone_GuardiansofFarm
```

2. **Create Virtual Environment**
```bash
conda create -n smartfarm python=3.8
conda activate smartfarm
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
```
ultralytics>=8.0.0
torch>=1.12.0
opencv-python>=4.6.0
numpy>=1.21.0
pillow>=9.0.0
albumentations>=1.3.0
```

### Running the Detection System

#### Basic Usage

1. **Prepare Input Data**
```bash
# Place images in the input directory
mkdir -p chocomint
cp /path/to/your/images/*.jpg chocomint/
```

2. **Execute Detection**
```bash
python detector.py
```

3. **View Results**
```bash
# Results automatically saved to chocomint_result/
ls chocomint_result/
```

#### Advanced Usage

**Custom Configuration:**
```python
# detector.py configuration options
config = {
    'model_path': 'weights/best.pt',
    'confidence_threshold': 0.25,
    'iou_threshold': 0.45,
    'input_dir': 'chocomint',
    'output_dir': 'chocomint_result',
    'save_json': True,
    'device': 'cuda'  # or 'cpu'
}
```

**Batch Processing:**
```bash
# Process entire directory
python detector.py --input_dir /path/to/images --batch_size 16

# GPU acceleration
python detector.py --device cuda --half  # FP16 inference
```

**Real-time Monitoring:**
```bash
# Webcam integration (future feature)
python detector.py --source webcam --stream
```

### Output Interpretation

Each processed image generates:

1. **Visual Output** (`*_output.jpg`)
   - Bounding boxes around detected leaves
   - Class labels (healthy/diseased)
   - Confidence scores
   - Overall health status banner

2. **Metadata File** (`*_metadata.json`)
```json
{
  "filename": "plant_001.jpg",
  "timestamp": "2024-12-18T10:30:45",
  "total_leaves": 24,
  "healthy_leaves": 20,
  "diseased_leaves": 4,
  "health_ratio": 83.33,
  "status": "NORMAL",
  "detections": [...]
}
```

---

## 📊 Performance Analysis

### Quantitative Metrics

#### Overall Performance

| Metric | Value | Industry Benchmark |
|--------|-------|-------------------|
| **mAP@0.5** | **0.810** | 0.75-0.85 |
| **mAP@0.5:0.95** | **0.592** | 0.50-0.65 |
| **Inference Time** | **45ms** | <100ms |
| **FPS** | **22** | >15 |
| **Model Size** | **6.2 MB** | <10 MB |

#### Class-Specific Performance

| Class | Precision | Recall | F1-Score | AP@0.5 |
|-------|-----------|--------|----------|--------|
| Healthy Leaf | 0.812 | 0.768 | 0.789 | **0.774** |
| Diseased Leaf | 0.891 | 0.823 | 0.856 | **0.846** |
| **Weighted Avg** | **0.852** | **0.796** | **0.823** | **0.810** |

### Qualitative Analysis

#### Confusion Matrix
```
                  Predicted
               Healthy  Diseased
Actual Healthy   3,245      412
       Diseased    298    2,987
```

**Key Insights:**
- **High True Positive Rate**: 91.2% for diseased leaf detection
- **Low False Negative Rate**: Critical for early disease intervention
- **Balanced Performance**: Minimal bias toward either class

#### Error Analysis

**Common Failure Modes:**
1. **Occlusion**: Overlapping leaves reduce detection accuracy (12% of errors)
2. **Lighting Variation**: Extreme shadows or highlights cause misclassification (8%)
3. **Early-Stage Disease**: Subtle symptoms harder to detect (15%)
4. **Boundary Cases**: Leaves at image edges sometimes missed (5%)

**Mitigation Strategies:**
- Enhanced augmentation for occlusion scenarios
- HDR image processing for lighting normalization
- Multi-temporal analysis for early disease detection
- Padding strategy for edge detection improvement

### Computational Efficiency

**Hardware Performance:**

| Hardware | Batch Size | Throughput (img/s) | Latency (ms) |
|----------|------------|-------------------|--------------|
| NVIDIA RTX 3090 | 1 | 22.2 | 45 |
| NVIDIA RTX 3090 | 16 | 285 | 56 (per batch) |
| Intel i7-12700K (CPU) | 1 | 3.8 | 263 |
| Jetson Xavier NX | 1 | 8.5 | 118 |

**Memory Footprint:**
- Model Weights: 6.2 MB
- Runtime Memory (GPU): ~1.2 GB
- Peak Memory (Training): ~8.5 GB

---

## 🎓 Research Contributions

### Academic Significance

This project demonstrates advanced competencies essential for graduate-level research in computer vision and multimodal AI:

#### 1. **End-to-End ML System Design**

- **Problem Formulation**: Translated agricultural challenges into tractable computer vision tasks
- **Data Engineering**: Designed comprehensive data collection and augmentation pipeline
- **Model Selection**: Justified architectural choices based on deployment constraints
- **Evaluation Methodology**: Established rigorous metrics aligned with domain requirements

#### 2. **Domain Adaptation Expertise**

- **Transfer Learning**: Successfully adapted general-purpose object detector to specialized agricultural domain
- **Few-Shot Learning Potential**: Framework supports rapid adaptation to new crop species with minimal additional data
- **Cross-Domain Generalization**: Model trained on diverse conditions demonstrates robust real-world performance

#### 3. **Practical AI Deployment**

- **Edge Computing Optimization**: Lightweight model suitable for resource-constrained farm environments
- **Real-Time Inference**: Achieved processing speeds enabling continuous monitoring applications
- **Interpretable Outputs**: Designed classification system accessible to non-technical end users

### Relevance to Vision-Language Models (VLM)

This project establishes a strong foundation for advanced VLM research:

#### **Current Capabilities → VLM Extensions**

| Current Feature | VLM Research Direction |
|----------------|------------------------|
| Visual leaf detection | **Visual question answering** about plant health |
| Health status classification | **Natural language disease descriptions** |
| Bounding box annotations | **Grounded language generation** with spatial reasoning |
| Multi-class detection | **Open-vocabulary disease recognition** |
| Temporal monitoring data | **Predictive text generation** for disease progression |

#### **Proposed VLM Research Directions**

1. **Multimodal Agricultural Assistant**
   - Enable farmers to query: *"Which plants need immediate attention?"*
   - Generate responses: *"Three plants in row 5 show early blight symptoms. Recommend fungicide treatment within 48 hours."*

2. **Few-Shot Disease Recognition**
   - Leverage vision-language pre-training for novel disease adaptation
   - Minimal labeled examples + textual descriptions → rapid model updates

3. **Knowledge-Grounded Recommendations**
   - Integrate agricultural domain knowledge with visual understanding
   - Generate contextualized treatment protocols based on detected conditions

4. **Explainable AI for Agriculture**
   - Natural language explanations for model predictions
   - Build farmer trust through transparent decision-making

### Publication & Presentation Potential

**Target Venues:**
- Computer Vision: CVPR, ICCV, ECCV (Workshops)
- Agriculture AI: AAAI AI for Agriculture, AgriVision
- Applied ML: NeurIPS Applications Track, ICML

**Contribution Areas:**
- Novel dataset for plant disease detection
- Lightweight model architecture for edge deployment
- Benchmark results for smart farm monitoring systems

---

## 🔮 Future Directions

### Immediate Enhancements (6-12 months)

#### 1. **Dataset Expansion**
- [ ] Multi-crop species support (tomatoes, peppers, cucumbers)
- [ ] Fine-grained disease classification (10+ disease types)
- [ ] Temporal datasets tracking disease progression
- [ ] Multi-spectral imaging (NIR, thermal)

#### 2. **Model Architecture Improvements**
- [ ] Attention mechanisms for fine-grained localization
- [ ] Transformer-based detectors (DETR variants)
- [ ] Multi-task learning (detection + segmentation)
- [ ] Uncertainty quantification for prediction confidence

#### 3. **System Integration**
- [ ] IoT sensor fusion (temperature, humidity, soil moisture)
- [ ] Edge deployment on agricultural drones
- [ ] Cloud-based farm management dashboard
- [ ] Mobile application for farmers

### Long-Term Research Vision (1-3 years)

#### **Vision-Language Model Integration**

**1. Agricultural VLM Development**
```
Project Goal: Develop domain-specific VLM for agriculture
├── Phase 1: Multimodal Pre-training
│   ├── Vision Encoder: Agricultural image encoder (ViT/ResNet)
│   ├── Language Encoder: Domain-adapted BERT/GPT
│   └── Contrastive Learning: Image-text alignment
├── Phase 2: Task-Specific Fine-tuning
│   ├── Visual Question Answering (VQA)
│   ├── Image Captioning with Technical Details
│   └── Grounded Disease Explanation
└── Phase 3: Deployment & Evaluation
    ├── User Studies with Farmers
    ├── Real-world Performance Metrics
    └── Continual Learning Framework
```

**2. Research Questions**

- **RQ1**: Can vision-language models improve disease detection accuracy through textual symptom descriptions?
- **RQ2**: How effective are few-shot learning approaches for rapidly adapting to novel crop diseases?
- **RQ3**: What role does agricultural domain knowledge play in grounding VLM outputs?
- **RQ4**: Can natural language interfaces increase technology adoption among farmers?

**3. Technical Innovations**

- **Hierarchical Vision-Language Alignment**: Align visual features at multiple granularities (pixel → region → image) with corresponding text descriptions
- **Knowledge Graph Integration**: Incorporate agricultural ontologies (disease taxonomy, treatment protocols) into VLM reasoning
- **Temporal Reasoning**: Extend VLMs to understand disease progression over time-series images
- **Active Learning**: Develop query strategies for efficient data collection guided by model uncertainty

#### **Precision Agriculture Platform**

**System Architecture:**
```
┌─────────────────────────────────────────────┐
│         Cloud Analytics Platform            │
│  ┌─────────────────────────────────────┐   │
│  │  Multi-Modal Data Fusion            │   │
│  │  • Vision Models                    │   │
│  │  • Language Models                  │   │
│  │  • Sensor Data Analytics            │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Predictive Models                  │   │
│  │  • Disease Risk Assessment          │   │
│  │  • Yield Forecasting                │   │
│  │  • Treatment Optimization           │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ▲
                    │ Data Upload
                    │
┌─────────────────────────────────────────────┐
│           Edge Devices (Farm Level)         │
│  ┌──────────────┐  ┌──────────────┐        │
│  │  Monitoring  │  │  Intervention│        │
│  │   Cameras    │  │    Drones    │        │
│  └──────────────┘  └──────────────┘        │
│  ┌──────────────┐  ┌──────────────┐        │
│  │  IoT Sensors │  │  Mobile App  │        │
│  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────┘
```

**Key Features:**
- Real-time monitoring across multiple farms
- Predictive analytics for disease outbreaks
- Automated treatment recommendations
- Natural language farmer interface
- Sustainability metrics tracking

### Broader Impact

**Scientific Contributions:**
- Advance state-of-the-art in agricultural AI
- Develop benchmarks for multimodal agricultural understanding
- Publish open-source datasets and models

**Societal Impact:**
- Increase global food security through disease prevention
- Reduce pesticide usage via precision intervention
- Empower smallholder farmers with accessible AI tools
- Support sustainable agriculture practices

---

## 🤝 Collaboration & Contribution

### Contributing Guidelines

We welcome contributions from the community! Areas of interest:

- **Data Collection**: Contribute labeled agricultural images
- **Model Improvements**: Propose architectural enhancements
- **Feature Requests**: Suggest new functionality
- **Bug Reports**: Identify and document issues

**Contribution Process:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes with descriptive messages
4. Push to your fork (`git push origin feature/YourFeature`)
5. Open a Pull Request with detailed description

### Research Collaboration

Interested in collaborating on VLM research or agricultural AI? I'm actively seeking opportunities to:
- Develop novel multimodal architectures for precision agriculture
- Publish in top-tier computer vision and AI conferences
- Build practical AI systems with real-world impact

---

## 📧 Contact

**Sewon Kim**  
Computer Science & Engineering Student  
Jeonbuk National University

- 📧 Email: sewonkim1018@gmail.com
- 🌐 Website: [andimsewon.github.io](https://andimsewon.github.io)
- 💼 LinkedIn: [linkedin.com/in/sewon-kim-742a492a6](https://www.linkedin.com/in/sewon-kim-742a492a6/)
- 🐙 GitHub: [github.com/andimsewon](https://github.com/andimsewon)

---

## 📚 References & Acknowledgments

### Key Technologies

- **YOLOv8**: Jocher, G., et al. (2023). Ultralytics YOLOv8. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **PyTorch**: Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.

### Datasets

- Plant Village Dataset (Hughes & Salathé, 2015)
- Kaggle Plant Disease Recognition Dataset
- Industry partner proprietary data (anonymized)

### Acknowledgments

Special thanks to:
- **Industry Partner**: For providing domain expertise and real-world data
- **Capstone Advisors**: For guidance throughout the project
- **Research Lab Colleagues**: For valuable feedback and support

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🏆 Project Recognition

**Achievements:**
- Selected for Industry-Academic Capstone Showcase (Fall 2024)
- Best Technical Implementation Award (Jeonbuk National University)
- Featured in University AI Research Symposium

---

<div align="center">

**Built with 🌱 for advancing AI in sustainable agriculture**

*This project represents a stepping stone toward developing interpretable, multimodal AI systems that bridge computer vision and natural language understanding—with the ultimate goal of creating accessible, impactful technologies for precision agriculture.*

---

### 🎯 Research Philosophy

*"The most impactful AI research solves real-world problems while pushing theoretical boundaries. This project embodies that principle: addressing urgent agricultural challenges while establishing foundations for next-generation vision-language models that can truly understand and communicate about the visual world."*

---

**⭐ If you find this work interesting, please consider starring the repository!**

[⬆ Back to Top](#smart-farm-leaf-detection-and-disease-identification-system)

</div>
