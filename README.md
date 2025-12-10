# Federated Agentic AI Platform for Predictive Vehicle Maintenance  
### Author: Akash Karri 
### EY Techathon 6.0 – Challenge 3

---

## 🚗 Project Overview

Modern vehicles produce large volumes of telematics data, yet breakdowns continue to occur due to limited real-time intelligence and centralized analytics constraints. OEMs also struggle with uneven service loads, inefficient scheduling, reactive customer engagement, and lack of manufacturing feedback loops.  
This project proposes a **Federated Agentic AI Platform** that delivers a complete, privacy-preserving, end-to-end predictive maintenance ecosystem.

The platform integrates:  
- **Federated Learning (FL)** for collaborative model training without sharing raw data  
- **Agentic AI Architecture** for autonomous diagnostics, communication, scheduling, forecasting, RCA/CAPA, and security  
- **Gemini LLM** for personalized, human-readable maintenance explanations  
- **Streamlit Dashboard** for interactive real-time demonstration  

This repository contains the working prototype submitted for Zone-2 of EY Techathon 6.0.

---

## 🧠 Key Features

### **1. Federated Learning–Powered Failure Prediction**
- Local vehicle nodes simulate decentralized telematics data.
- Each node trains a model locally using **SGDClassifier**.
- **FedAvg aggregation** produces a global predictive model.
- AUC progression is plotted across rounds.

### **2. Agentic AI System**
The system is orchestrated by a **Master Agent**, with specialized Worker Agents:

- **Telemetry Analysis Agent**  
- **Failure Prediction Agent**  
- **Communication Agent (Gemini-LangChain based)**  
- **Demand Forecasting Agent**  
- **Service Scheduling Agent**  
- **Service Lifecycle Agent**  
- **RCA / CAPA Insights Agent**  
- **UEBA Security Agent**

### **3. Real-Time Dashboard (Streamlit)**
- Adjust telematics parameters using sliders  
- Get failure probability and classification  
- Component-wise diagnostic breakdown (Engine, Cooling, Suspension, Usage Fatigue)  
- Trend visualization (simulated last 10 trips)  
- FL AUC-vs-Rounds chart  
- Gemini-powered explanation for recommended action  

### **4. Privacy, Security & Compliance**
- No raw telematics data ever leaves the node  
- UEBA-based agent behavior monitoring  
- Modular, scalable multi-agent architecture

---

## 🏗️ System Architecture

The architecture consists of:

- **Vehicle Nodes (FL Clients):** Perform local training on synthetic telemetry  
- **Federated Learning Server:** Aggregates model weights -> global model  
- **Master Orchestrator Agent:** Central controller  
- **Worker Agents:** Execute tasks based on risk, telemetry, and system conditions  
- **Owner Interface:** Voice or mobile notification (concept)  
- **Dashboards:** Maintenance, forecasting, RCA  

(Architecture diagram included in PPT submission)

---

## 🔧 Tech Stack

| Layer | Tools / Technologies |
|-------|----------------------|
| Programming | Python 3.10 |
| ML Frameworks | Scikit-learn, Flower (Federated Learning) |
| Dashboard | Streamlit |
| LLM | Google Gemini Pro |
| Data | Synthetic telematics simulator |
| Visualization | Matplotlib, Streamlit native charts |
| Security | UEBA conceptual framework |
| Deployment | Local / modular microservice-ready |

---

## 📁 Folder Structure

```bash
├── data/
│   └── simulated_vehicle_data.csv
├── models/
│   ├── global_model.pkl
│   └── fl_metrics.pkl
├── generate_data.py
├── federated_training.py
├── streamlit_app.py
├── README.md
├── requirements.txt
