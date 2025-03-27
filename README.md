# 🫁 Quantum AI Pneumonia Detection

This project is an **end-to-end AI solution** that integrates **Quantum AI, Computer Vision, and Quantum Model** to detect **Pneumonia from Chest X-ray scans**. It utilizes **Quantum Support Vector Classifiers (QSVC)** for enhanced medical diagnostics.

---

## 🚀 Features

- 🧠 Automated Chest X-ray Preprocessing & Augmentation  
- 🔬 Quantum Computation  
- ☁️ Aersimulator for Quantum Feature Extraction  
- 🖼️ Real-Time Web UI for X-ray Visualization & Quantum Prediction  
- 💻 Command-Line Interface (CLI) for Direct Model Execution  
- 🐳 Fully Dockerized for Seamless Deployment  
- 📤 Upload Chest X-ray Images via Web Interface (image size 256x256)

---

## 🏗 Solution Architecture

### 🔬 End-to-End Processing Pipeline

1. **Chest X-ray Preprocessing**  
   - Loads **Pneumonia & Normal** X-ray images from datasets  
   - Applies grayscale conversion, Gaussian blur, and scaling  

2. **Quantum Feature Extraction**  
   - Reduces dimensionality with **PCA**  
   - Encodes features into a **Quantum Circuit Ansatz**

3. **Quantum Model Training & Classification**  
   - Trains **Quantum Support Vector Classifiers (QSVC)** 
   - Leverages quantum learning for robust pneumonia detection

4. **Workflow Automation**  
   - Uses **JobScheduler** and **WorkflowManager** to manage execution  
   - Submits jobs to **Quantum Runtime** backend

5. **Real-Time Visualization & Prediction**  
   - Web interface enables:
     - 📤 Uploading Chest X-rays  
     - 📈 Viewing PCA plots and probability histograms  
     - 🧠 Running quantum model predictions

---

## 🖼 Upload Chest X-ray via Web UI

### ✅ What you can do:

- Select or drag-and-drop a Chest X-ray image (`.jpg`, `.png`, `.jpeg`)  
- Get **immediate predictions** using local simulation or  
- Submit to **Quantum** for sync real-backend inference  
- Polls IBM for results automatically, displays them in real time  
- Visualizes prediction result, probability, and relevant plots

### 💡 How it works:

1. Image is **grayscaled**, blurred, and resized  
2. PCA reduction → 18 features  
3. Expanded into 54 quantum circuit params  
4. Run via **PegasosQSVC**
5. Prediction shown in browser after processing  

---

## 🏗 Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-repo/Quantum-Pneumonia-Detection
cd Quantum-Pneumonia-Detection
```

### 2️⃣ **Setup Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # MacOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🔥 Running the System

### **1️⃣ CLI Mode**
```bash
python interfaces/cli.py --dataset-info
python interfaces/cli.py --model-score
python interfaces/cli.py --predict
```
✅ **Output Example:**  
`Quantum QSVC on the training dataset: 0.75`
`Quantum QSVC on the test dataset: 0.68`

---

### **2️⃣ Web Interface**
```bash
python interfaces/web_app/app.py
```
🖥 **Access Web App:**  
🔍 Open **`http://127.0.0.1:5000/`** in a browser.
`📌 Web UI Features:`
✅` 📤 Upload Chest X-rays.`
✅ `🔬 Run local for IBM Quantum classification`
✅ `📊 View PCA plots, confusion matrix, and prediction histograms`

---

## 🐳 Deploying with Docker

### **1️⃣ Build Docker Image**
```bash
docker build -t quantum-pneumonia .
```

### **2️⃣ Run Container**
```bash
docker run -p 5000:5000 quantum-pneumonia

if echo "QISKIT_IBM_TOKEN=your_ibm_quantum_token_here" > .env
docker run --env-file .env -p 5000:5000 quantum-pneumonia
```

🖥 **Access Web App:**  
🔍 Open **`http://127.0.0.1:5000/`**

---

## 🛠️ Development & Testing

### **Run PyTests**
```bash
pytest tests/
```

---

## 💼 IBM Quantum Cloud Integration

**Setup IBM Quantum Account**  
1. Create an account at [IBM Quantum](https://quantum-computing.ibm.com/)
2. Get your API **Token** from **My Account**
3. Set it in your environment:
```bash
export QISKIT_IBM_TOKEN="your_ibm_quantum_token"
```

---
