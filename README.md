# MLOps-Mini-Project: Emotion Detection ( Happy or sad )

[![CI Pipeline](https://github.com/Pravat-21/MLops-Mini-Project/actions/workflows/ci.yaml/badge.svg)](https://github.com/Pravat-21/MLops-Mini-Project/actions/workflows/ci.yaml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-orange.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)

An end-to-end MLOps pipeline for emotion classification from tweets, implementing industry-standard practices including experiment tracking, model versioning, continuous integration/deployment, and containerization.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Tracking](#monitoring--tracking)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete MLOps pipeline for classifying tweet emotions (specifically happiness vs. sadness) using natural language processing and machine learning. The pipeline incorporates modern DevOps practices, reproducible workflows, and automated deployment.

### Key Objectives:
- **Reproducibility**: Version control for data, code, and models using DVC
- **Automation**: CI/CD pipeline with GitHub Actions
- **Experiment Tracking**: MLflow integration with DagsHub
- **Containerization**: Docker for consistent deployment
- **Monitoring**: Comprehensive logging and metrics tracking
- **Production-Ready**: Deployed Flask application with model registry

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Data Source (GitHub)                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Data Ingestion & Split                          │
│                    (Train/Test: 80/20 split)                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Preprocessing                                │
│              (Text Cleaning, Normalization)                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Feature Engineering                                │
│        (Bag of Words - max_features: 5500)                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Model Building                                  │
│                 (Logistic Regression)                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Model Evaluation                                   │
│            (Metrics Generation & Logging)                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Model Registration                                  │
│               (MLflow Model Registry)                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Flask Web Application                              │
│                (Containerized with Docker)                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Deployment (Render)                               │
└─────────────────────────────────────────────────────────────────────┘
```

## ✨ Features

### MLOps Best Practices
- ✅ **Version Control**: Git for code, DVC for data and models
- ✅ **Experiment Tracking**: MLflow with DagsHub integration
- ✅ **Pipeline Orchestration**: DVC pipelines with dependency management
- ✅ **Model Registry**: Centralized model versioning and staging
- ✅ **Continuous Integration**: Automated testing and validation
- ✅ **Continuous Deployment**: Automated Docker build and deployment
- ✅ **Containerization**: Docker for consistent environments
- ✅ **Configuration Management**: YAML-based parameter configuration
- ✅ **Logging**: Comprehensive logging at every stage
- ✅ **Testing**: Unit tests for model and application

### Machine Learning Pipeline
- 📊 Data ingestion from remote source
- 🧹 Text preprocessing and cleaning
- 🔤 Feature engineering with Bag of Words
- 🤖 Model training with hyperparameter configuration
- 📈 Automated metrics tracking
- 🎯 Model evaluation and registration
- 🚀 Production deployment with Flask

## 📁 Project Structure

```
MLOps-Mini-Project/
│
├── .dvc/                       # DVC configuration and cache
├── .github/                    # GitHub configurations
│   └── workflows/
│       └── ci.yaml            # CI/CD pipeline configuration
│
├── data/                       # Data directory (DVC tracked)
│   ├── external/              # External data sources
│   ├── interim/               # Intermediate processed data
│   ├── processed/             # Final feature-engineered data
│   └── raw/                   # Raw data (train/test split)
│
├── docs/                       # Documentation files
│
├── flask_app/                  # Flask web application
│   ├── templates/             # HTML templates
│   ├── app.py                 # Flask application
│   ├── preprocessing_utility.py  # Text preprocessing utilities
│   └── requirements.txt       # Flask app dependencies
│
├── logs/                       # Application logs
│
├── models/                     # Trained models (DVC tracked)
│   ├── model.pkl              # Trained classifier
│   └── vectorizer.pkl         # Fitted vectorizer
│
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── 01exp_basemodel.ipynb
│   ├── 02exp_bowVStfidf.ipynb
│   └── 03exp_logistic_Hyper_tuning.ipynb
│
├── references/                 # Reference materials and documentation
│
├── reports/                    # Generated reports and metrics
│   ├── metrics.json           # Model performance metrics
│   └── experiment_info.json   # Experiment metadata
│
├── scripts/                    # Utility scripts
│   └── promote_model.py       # Model promotion script
│
├── src/                        # Source code
│   ├── data/                  # Data processing modules
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   │
│   ├── features/              # Feature engineering
│   │   └── feature_engineering.py
│   │
│   ├── model/                 # Model training and evaluation
│   │   ├── model_building.py
│   │   ├── model_evaluation.py
│   │   └── register_model.py
│   │
│   ├── visualization/         # Visualization utilities
│   │   └── visualize.py
│   │
│   ├── exception.py           # Custom exception handling
│   ├── logger.py              # Logging configuration
│   └── utils.py               # Utility functions
│
├── tests/                      # Test suite
│   ├── test_model.py          # Model tests
│   └── test_flask_app.py      # Application tests
│
├── .dvcignore                  # DVC ignore patterns
├── .env                        # Environment variables (not tracked)
├── .gitignore                  # Git ignore patterns
├── Dockerfile                  # Docker container configuration
├── dvc.lock                    # DVC pipeline lock file
├── dvc.yaml                    # DVC pipeline definition
├── LICENSE                     # Project license
├── Makefile                    # Build automation
├── params.yaml                 # Pipeline parameters
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── test_environment.py         # Environment validation
└── tox.ini                     # Tox testing configuration
```

## 🛠️ Tech Stack

### Machine Learning & Data Science
- **Python 3.10**: Core programming language
- **scikit-learn**: Machine learning algorithms
- **pandas & numpy**: Data manipulation
- **NLTK**: Natural language processing

### MLOps Tools
- **DVC**: Data version control and pipeline orchestration
- **MLflow**: Experiment tracking and model registry
- **DagsHub**: Remote storage and collaboration platform

### Development & Deployment
- **Flask**: Web application framework
- **Docker**: Containerization
- **Gunicorn**: WSGI HTTP server
- **GitHub Actions**: CI/CD automation
- **Render**: Cloud deployment platform

### Supporting Tools
- **Git**: Version control
- **pytest**: Testing framework
- **Makefile**: Build automation

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Git
- Docker (optional, for containerization)
- DVC (optional, for data versioning)

### Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/Pravat-21/MLops-Mini-Project.git
cd MLOps-Mini-Project
```

2. **Create Virtual Environment**
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Package in Development Mode**
```bash
pip install -e .
```

5. **Download NLTK Data**
```bash
python -m nltk.downloader stopwords wordnet
```

6. **Set Up Environment Variables**
Create a `.env` file in the root directory:
```env
DAGSHUB_PAT=your_dagshub_personal_access_token
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
```

7. **Initialize DVC** (if not already initialized)
```bash
dvc init
```

## 💻 Usage

### Running the Complete Pipeline

#### Using DVC (Recommended)
```bash
# Run the entire pipeline
dvc repro

# Run specific stage
dvc repro data_ingestion
```

#### Using Make
```bash
# Install dependencies
make requirements

# Run data processing
make data

# Clean compiled files
make clean
```

#### Manual Execution
```bash
# Run individual stages
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
python src/features/feature_engineering.py
python src/model/model_building.py
python src/model/model_evaluation.py
python src/model/register_model.py
```

### Running the Flask Application

#### Local Development
```bash
cd flask_app
python app.py
```
Access the application at `http://localhost:5000`

#### Using Docker
```bash
# Build Docker image
docker build -t emotion-classifier:latest .

# Run container
docker run -p 5000:5000 emotion-classifier:latest
```

## 🔄 Pipeline Stages

The DVC pipeline consists of six interconnected stages defined in `dvc.yaml`:

### 1. Data Ingestion
- **Command**: `python src/data/data_ingestion.py`
- **Dependencies**: Source code
- **Parameters**: `test_size: 0.20`
- **Outputs**: `data/raw/` directory
- **Description**: Downloads tweet emotions dataset, filters for happiness/sadness, and splits into train/test sets

### 2. Data Preprocessing
- **Command**: `python src/data/data_preprocessing.py`
- **Dependencies**: Raw data, source code
- **Outputs**: `data/interim/` directory
- **Description**: Text cleaning including:
  - Lowercase conversion
  - Stop words removal
  - Number removal
  - Punctuation removal
  - URL removal
  - Lemmatization

### 3. Feature Engineering
- **Command**: `python src/features/feature_engineering.py`
- **Dependencies**: Interim data, source code
- **Parameters**: `max_features: 5500`
- **Outputs**: 
  - `data/processed/` directory
  - `models/vectorizer.pkl`
- **Description**: Transforms cleaned text into numerical features using Bag of Words

### 4. Model Building
- **Command**: `python src/model/model_building.py`
- **Dependencies**: Processed data, source code
- **Outputs**: `models/model.pkl`
- **Description**: Trains Logistic Regression classifier with optimized hyperparameters

### 5. Model Evaluation
- **Command**: `python src/model/model_evaluation.py`
- **Dependencies**: Trained model, source code
- **Metrics**: `reports/metrics.json`
- **Outputs**: `reports/experiment_info.json`
- **Description**: Evaluates model performance and logs metrics to MLflow

### 6. Model Registration
- **Command**: `python src/model/register_model.py`
- **Dependencies**: Experiment info, source code
- **Description**: Registers model in MLflow Model Registry for version control

## 🎓 Model Training

### Configuration
Model parameters are defined in `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.20

feature_engineering:
  max_features: 5500

model_building:
  n_estimators: 55
  learning_rate: 0.1
```

### Algorithm
The project uses **Logistic Regression** with the following configuration:
- **Regularization**: L2 (Ridge)
- **Solver**: liblinear
- **C Parameter**: 1.0

### Experiment Tracking
All experiments are tracked using MLflow with DagsHub integration:
- Training metrics
- Model parameters
- Model artifacts
- Run metadata

Access experiments at: `https://dagshub.com/Pravat-21/MLops-Mini-Project.mlflow`

## 🌐 Deployment

### Docker Deployment

The application is containerized using Docker for consistent deployment across environments.

**Dockerfile Configuration:**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
```

**Build and Run:**
```bash
# Build image
docker build -t emotion-classifier:latest .

# Run container
docker run -p 5000:5000 \
  -e DAGSHUB_PAT=your_token \
  emotion-classifier:latest
```

### Cloud Deployment (Render)

The application is automatically deployed to Render through the CI/CD pipeline:

1. GitHub Actions builds Docker image
2. Image is pushed to Docker Hub
3. Render deployment is triggered via API
4. Application is available at production URL

## 🔁 CI/CD Pipeline

The project implements a comprehensive CI/CD pipeline using GitHub Actions (`.github/workflows/ci.yaml`):

### Pipeline Stages

```yaml
1. Code Checkout
   ↓
2. Python Setup (3.10)
   ↓
3. Dependency Caching
   ↓
4. Install Dependencies
   ↓
5. Run DVC Pipeline
   ↓
6. Model Testing
   ↓
7. Model Promotion (if tests pass)
   ↓
8. Flask App Testing
   ↓
9. Docker Login
   ↓
10. Build Docker Image
   ↓
11. Push to Docker Hub
   ↓
12. Trigger Render Deployment
```

### Trigger
The pipeline runs on every push to the repository.

### Required Secrets
Configure these secrets in GitHub repository settings:
- `DAGSHUB_PAT`: DagsHub personal access token
- `DOCKER_HUB_USERNAME`: Docker Hub username
- `DOCKER_HUB_ACCESS_TOKEN`: Docker Hub access token
- `RENDER_SERVICE_ID`: Render service ID
- `RENDER_API_KEY`: Render API key

## 📊 Monitoring & Tracking

### Logging
Comprehensive logging is implemented across all modules:
- **File Logging**: Detailed logs in `logs/` directory
- **Console Logging**: Real-time feedback during execution
- **Custom Exception Handling**: Detailed error tracking

### Metrics Tracking
Performance metrics are automatically tracked and stored:
- **Accuracy**: Overall model accuracy
- **Precision**: Precision score
- **Recall**: Recall score
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Classification results

Metrics are stored in `reports/metrics.json` and logged to MLflow.

### Experiment Management
MLflow provides:
- Experiment comparison
- Model versioning
- Artifact storage
- Parameter tracking
- Metric visualization

## 📝 API Documentation

### Endpoints

#### GET `/`
- **Description**: Home page with prediction form
- **Response**: HTML page

#### POST `/predict`
- **Description**: Predict emotion from text input
- **Request Body**:
  ```json
  {
    "text": "Your tweet text here"
  }
  ```
- **Response**: HTML page with prediction result
  - `0`: Sadness
  - `1`: Happiness

### Example Usage

**Using cURL:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=I am feeling great today!"
```

**Using Python:**
```python
import requests

url = "http://localhost:5000/predict"
data = {"text": "I am feeling great today!"}
response = requests.post(url, data=data)
print(response.text)
```

## 🧪 Testing

### Test Structure
Tests are organized in the `tests/` directory:
- `test_model.py`: Model functionality tests
- `test_flask_app.py`: Application endpoint tests

### Running Tests

**Using unittest:**
```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests/test_model.py
```

**Using pytest:**
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Test Coverage
Tests cover:
- Data preprocessing functions
- Model training and prediction
- Flask API endpoints
- Utility functions
- Exception handling

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make Changes** and commit with clear messages
   ```bash
   git commit -m "Add: description of your feature"
   ```
4. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Create a Pull Request**

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation as needed

## 📄 License

This project is open-source and available under the terms specified in the LICENSE file.

## 👤 Author

**Pravat Patra**

- GitHub: [@Pravat-21](https://github.com/Pravat-21)
- DagsHub: [MLops-Mini-Project](https://dagshub.com/Pravat-21/MLops-Mini-Project)

## 🙏 Acknowledgments

- Project template based on [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- Dataset source: CampusX Jupyter Masterclass
- MLOps practices inspired by industry standards

## 📚 Resources

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 🚀 Quick Start

Get started in 5 minutes:

```bash
# Clone the repository
git clone https://github.com/Pravat-21/MLops-Mini-Project.git
cd MLOps-Mini-Project

# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run the pipeline
dvc repro

# Start the application
cd flask_app
python app.py
```

Visit `http://localhost:5000` and start classifying emotions! 🎉

---

**Note**: Make sure to set up your environment variables and DagsHub credentials before running the pipeline.

For detailed instructions, refer to the [Installation](#installation) section.
