# 📞 Azure Call Center Triage Dashboard

## 📖 Overview
The **Call Center Triage Dashboard** is an enterprise-grade **Proof of Concept (POC)** designed to automate the Quality Assurance (QA) process for customer service calls.

Authorized users can upload batches of audio files (MP3/WAV). The system asynchronously transcribes the audio using **Azure Speech Services** (with **Speaker Diarization**) and then grades the interaction using **Azure AI Prompt Flow** for consistency in grading, utilizing various openai models available, primarily targetting gpt5, gpt5 mini.

The result is a **Triage View** that automatically flags calls requiring management attention based on **low scores**, **negative sentiment**, or **unresolved issues**.

---

## 📑 Table of Contents
- Architecture
- Prerequisites
- Configuration & Environment Variables
- Deployment Guide  
  - Prompt Flow Setup  
  - Web App Deployment  
  - Security Setup (SSO)  
- Prompt Engineering
- Requirements
- Troubleshooting
- License

---

## 🏗️ Architecture

**Frontend**  
- Streamlit Web App (Python)
- Hosted on Azure App Service (Linux)

**Storage**  
- Azure Blob Storage (temporary audio storage)

**Transcription Engine**  
- Azure AI Speech (Batch Transcription API v3.2)
- Speaker Diarization
- MP3 & long‑audio support

**Intelligence Engine**  
- Azure AI Foundry (Prompt Flow)
- Model: GPT-5 or GPT-5-Mini (or chosen models)

**Security**  
- Azure App Service Authentication (Microsoft Entra ID)

---

## ⚡ Prerequisites

- Azure Subscription
- Python 3.10+

### Required Azure Resources
- Storage Account (Standard GPv2)
- Speech Service (Standard S0)
- Azure AI Foundry Hub & Project
- App Service (Linux, Python 3.10, B1 recommended)

---

## ⚙️ Configuration & Environment Variables

| Variable | Description |
|--------|-------------|
| SPEECH_KEY | Azure Speech API Key |
| SPEECH_REGION | Azure Region |
| STORAGE_CONN_STR | Blob Storage Connection String |
| PF_ENDPOINT | Prompt Flow Endpoint |
| PF_KEY | Prompt Flow API Key |

---

## 🚀 Deployment Guide

### 1. Prompt Flow Setup
- Create Standard Flow
- Input: transcript_text (String)
- Output: ${LLM.output}
- Deploy as real‑time endpoint

### 2. Web App Deployment
```bash
az webapp up --name <app-name> --resource-group <rg>
```

Startup Command:
```bash
pip install -r requirements.txt && python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
```

### 3. Security Setup
- App Service → Authentication
- Microsoft Provider
- Single Tenant
- Require authentication

---

## 🧠 Prompt Engineering
LLM must return **strict JSON only**:
```json
{
  "summary": "Two sentence summary",
  "questions": ["Customer questions"],
  "score": 0,
  "flags": []
}
```

---

## 📦 Requirements
```text
streamlit
requests
azure-storage-blob
pandas
openpyxl
```

---


---

## 📄 License
MIT License
