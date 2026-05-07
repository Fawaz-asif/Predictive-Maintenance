<div align="center">

# Predictive Maintenance Dashboard

### Real-Time Industrial Sensor Analytics & Failure Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=for-the-badge)](https://gradio.app/)
[![Deployed on Hugging Face](https://img.shields.io/badge/Deployed-Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](#)

### 🌐 **[Live Demo: Try the Dashboard on Hugging Face!](https://huggingface.co/spaces/fawazasif/Predictive-Maintenance)**

A complete machine learning pipeline and interactive dashboard that predicts industrial machine failures before they happen. Uses Decision Tree classification to analyze sensor data (rotational speed, torque, tool wear, temperature) and identify specific failure modes.

</div>

---

## Features

- **Real-Time Prediction**: Instantly predicts whether a machine will fail based on current sensor readings.
- **Failure Mode Classification**: Doesn't just say "Fail" -- identifies the specific *type* of failure:
  - Heat Dissipation Failure (HDF)
  - Power Failure (PWF)
  - Overstrain Failure (OSF)
  - Tool Wear Failure (TWF)
- **Batch Processing**: Upload a CSV of sensor logs to predict failures across hundreds of machines simultaneously.
- **Interactive UI**: Clean, modern web interface built with Gradio 6, featuring dynamic themes and data visualization.

---

## Architecture

1. **Preprocessing Pipeline** (pipeline.py): Cleans data, handles SMOTE balancing for rare failure types, and scales numerical features.
2. **Model Router** (outer.py): Implements a two-stage routing architecture. Stage 1 detects if a failure will occur. If yes, Stage 2 identifies the specific failure mode.
3. **Web Application** (pp.py): The Gradio frontend that handles user input, communicates with the router, and displays visually color-coded results.
4. **Hugging Face Deployment** (deploy_to_hf.py): Automated deployment scripts for Hugging Face Spaces.

---

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository:
\\\ash
git clone https://github.com/Fawaz-asif/Predictive-Maintenance.git
cd Predictive-Maintenance
\\\

2. Install dependencies:
\\\ash
pip install -r requirements.txt
\\\

3. Run the dashboard:
\\\ash
python app.py
\\\
*The app will be available at http://localhost:7860.*

---

## CI/CD

This repository uses **GitHub Actions** for continuous integration. On every push, the workflow automatically:
- Sets up the Python environment.
- Installs dependencies.
- Runs syntax checks and linting to ensure production-ready code.

---

## Dataset

The model is trained on the AI4I 2020 Predictive Maintenance Dataset (Predictive_M.csv).

## Author
**Fawaz Asif**