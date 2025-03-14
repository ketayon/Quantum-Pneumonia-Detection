# 🫁 Quantum AI Pneumonia Detection

This project is an **end-to-end AI solution** that integrates **Quantum AI, Computer Vision, and Hybrid Quantum-Classical Models** to detect **Pneumonia from Chest X-ray scans**. It utilizes **Quantum Kernel Learning (QKL)** and **Quantum Support Vector Classifiers (QSVC)** for enhanced medical diagnostics.

## 🚀 Features

- **Automated Chest X-ray Preprocessing & Augmentation**
- **Quantum-Classical Hybrid Computation**
- **IBM Quantum Cloud for Quantum Feature Extraction**
- **Real-Time Web UI for Chest X-ray Visualization & Prediction**
- **Command-Line Interface (CLI) for Direct Model Execution**
- **Fully Dockerized for Seamless Deployment**

---

## 🏗 **Solution Architecture**

### 🔬 **End-to-End Processing Pipeline**
1. **Chest X-ray Preprocessing**  
   - Loads **Pneumonia & Normal** X-ray images from datasets.  
   - Applies grayscale conversion & contrast enhancement.  

2. **Quantum Feature Extraction**  
   - Reduces X-ray data dimensionality using **PCA**.  
   - Encodes optimized data into **Quantum Kernel Circuits**.  

3. **Quantum Model Training & Classification**  
   - Uses **Quantum Support Vector Classifiers (QSVC)** trained on **IBM Quantum Cloud**.  
   - Hybrid **Quantum + Classical ML** improves Pneumonia detection accuracy.  

4. **Automated Workflow Execution**  
   - **JobScheduler & WorkflowManager** distribute quantum-classical computations.  
   - IBM **Quantum Backend** executes quantum-enhanced feature processing.  

5. **Real-Time Visualization & Prediction**  
   - **Web UI** provides **live Chest X-ray visualization**.  
   - Users can **upload scans, analyze quantum model predictions, and visualize probability heatmaps**.  

---

## 🏗 **Installation Guide**

### **1️⃣ Clone the Repository**
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
python interfaces/cli.py --model-score
```
✅ **Output Example:**  
`Quantum QSVC on the training dataset: 0.89`
`Quantum QSVC on the test dataset: 0.82`

---

### **2️⃣ Web Interface**
```bash
python interfaces/web_app/app.py
```
🖥 **Access Web App:**  
🔍 Open **`http://127.0.0.1:5000/`** in a browser.
`📌 Web UI Features:`
✅` Upload and analyze Chest X-ray scans.`
✅ `View Quantum Model Predictions.`
✅ `Visualize X-ray Images & Quantum Probability Heatmaps.`

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
