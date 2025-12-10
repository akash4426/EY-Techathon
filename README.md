# 🚗 Federated Agentic AI Platform for Predictive Vehicle Maintenance

### Author: Akash Karri  
### EY Techathon 6.0 – Challenge 3

---

## 🌟 Project Vision

Modern vehicles generate vast telematics data streams, yet unexpected breakdowns persist due to limitations in real-time insight and centralized analytics. OEMs also struggle with inconsistent maintenance outcomes and privacy hurdles.

**This platform introduces a breakthrough: A privacy-preserving, agentic AI ecosystem for predictive maintenance, powered by Federated Learning and multi-agent orchestration.**

---

## 🧩 Solution Highlights

- **Federated Learning:** Collaborative model training at the edge—no raw data leaves the vehicle!
- **Agentic AI System:** Autonomous agents diagnose, forecast, communicate, schedule, and secure.
- **Gemini LLM Integration:** Human-readable explanations tailored for vehicle owners.
- **Interactive Streamlit Dashboard:** Real-time simulation, insights, and recommendations.

*This repository showcases the working prototype submitted for Zone-2, EY Techathon 6.0.*

---

## 🚀 Key Features

### 1. Predictive Maintenance via Federated Learning
- Local nodes simulate decentralized vehicle data and train ML models.
- Global model aggregated via FedAvg; performance tracked across rounds.
- AUC chart visualizes learning progress.

### 2. Multi-Agent AI Architecture
Orchestrated by a **Master Agent**, with dedicated Worker Agents for:
- Telemetry Analysis
- Failure Prediction
- Owner Communication (Gemini-LangChain)
- Demand Forecasting
- Service Scheduling
- Lifecycle Management
- RCA/CAPA Insights
- UEBA Security Surveillance

### 3. Real-Time Dashboard
- Sliders for telematics input simulation
- Instant failure probability & classification
- Diagnostics: Engine, Cooling, Suspension, Usage Fatigue
- Trends: Last 10 simulated trips
- Federated Learning progress chart
- Gemini-powered recommendations

### 4. Privacy, Security & Compliance
- Raw data remains local—privacy by design
- Continuous behavioral monitoring via UEBA agent
- Modular, scalable architecture ready for expansion

---

## 🏗️ System Architecture

- **Vehicle Nodes (FL Clients):** Local training on synthetic telemetry
- **FL Server:** Aggregates weights → global model
- **Master Orchestrator:** Central controller of agents
- **Worker Agents:** Specialized tasks per system state
- **Owner Interface:** Concept for notifications (voice/mobile)
- **Dashboards:** Maintenance, forecasting, RCA
- *(Architecture diagram available in PPT submission)*

---

## 🛠️ Tech Stack

| Layer         | Tools / Technologies                 |
|---------------|-------------------------------------|
| Programming   | Python 3.10                         |
| ML Frameworks | Scikit-learn, Flower (FL)           |
| Dashboard     | Streamlit                           |
| LLM           | Google Gemini Pro                   |
| Data          | Synthetic telemetry simulator        |
| Visualization | Matplotlib, Streamlit charts         |
| Security      | UEBA conceptual framework           |
| Deployment    | Local / microservice-ready           |

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
```

---

## 📞 Contact & Credits

*For any queries or collaborations, reach out to Akash Karri.*

---

## 🏆 Awards & Recognition

- Prototype submitted for EY Techathon 6.0, Zone-2.

---

## 💡 Future Directions

- Expand agentic architecture for EVs and commercial fleets.
- Integrate advanced anomaly detection and reinforcement learning.
- Real-world pilot with OEM partners.

---
