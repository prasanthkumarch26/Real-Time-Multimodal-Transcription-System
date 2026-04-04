# 🎙️ Real-Time Multimodal Transcription System

A high-performance real-time transcription system for **live captions and streaming applications**, supporting both **speech** and **Indian Sign Language (ISL)**.

---

## 🚀 Key Results

- ⚡ **150+ chunks/sec throughput**
- ⏱️ **<300ms P99 latency**
- 🔁 **143ms recovery time after failures**
- ❄️ **Cold start reduced: 2.8s → 120ms**
- 📉 **~40% compute savings via Voice Activity Detection (VAD)**
- 📦 **~50MB memory per stream (O(1) bounded memory)**
- 📡 **~96 kbps bandwidth per stream**
- 💰 **~$0.002 per stream-minute (CPU-only inference)**

---

## 🧠 Why This Project Matters

Real-time transcription systems are difficult to build due to:

- Strict latency requirements (<500ms)
- Continuous streaming (unbounded memory risk)
- High concurrency and scaling challenges
- Handling silence, noise, and partial inputs

This system solves these using **asynchronous pipelines, intelligent buffering, backpressure control, and fault-tolerant design**.

---

## 🏗️ Architecture Overview

- **Frontend:** Next.js (real-time UI, streaming input)
- **Backend:** FastAPI + WebSockets (async streaming)
- **Audio Pipeline:** VAD → Whisper inference → streaming output
- **ISL Pipeline:** Frame extraction → LSTM → prediction
- **Design:** Stateless microservices for horizontal scaling

### Flow:
Client → WebSocket → Async Queue → Worker Pool → Model Inference → Real-time Response

---

## ⚡ Core Features

- **Real-time streaming transcription** via WebSockets  
- **Sliding window context handling** for accurate decoding  
- **Voice Activity Detection (VAD)** to skip silence  
- **Multimodal support** (speech + sign language)  
- **Stateless architecture** for resilience and scaling  
- **Backpressure & load shedding** for stability under overload  

---

## ⚙️ Key Engineering Decisions

### ⚡ Real-Time Inference Optimization

**Problem:** Standard Whisper models are too slow for real-time use  

**Solution:**
- Used `faster-whisper` with INT8 quantization  
- Optimized chunk-based streaming pipeline  

**Impact:**
- Real-Time Factor: **0.037**
- Sustained real-time transcription across concurrent streams  

---

### 🧠 Memory Optimization (O(1) Streaming)

**Problem:** Continuous streams cause unbounded memory growth  

**Solution:**
- Sliding window buffer  
- Aggressive silence removal using VAD  

**Impact:**
- Constant memory usage (~50MB per stream)  
- ~40% reduction in unnecessary compute  

---

### 🔁 Fault Tolerance & Recovery

**Problem:** Network drops break real-time sessions  

**Solution:**
- Stateless backend design  
- Client-side retry with reconnection + replay window  

**Impact:**
- **143ms recovery time**  
- **0% data loss during testing**  

---

### 📈 Load Testing & Bottleneck Analysis

**What was built:**
- Custom async load testing framework  

**Insights:**
- Identified blocking behavior in synchronous ASGI loops  
- Quantified saturation point at ~150 chunks/sec  

---

### 🌐 WebSocket-Based Streaming

**Why:**
- HTTP introduces repeated connection overhead  

**Solution:**
- Persistent WebSocket connections  

**Impact:**
- Reduced per-chunk latency by ~50–100ms  
- Enabled real-time bidirectional communication  

---

### 🔄 Backpressure & Load Shedding

**Problem:** Systems collapse under overload due to unbounded queues  

**Solution:**
- Bounded `asyncio.Queue` with tail-drop policy  
- Adaptive throttling based on queue pressure  

**Impact:**
- Stable latency even beyond saturation  
- Prevented cascading failures across streams  

---

### ⚙️ Concurrency & Event Loop Optimization

**Problem:** CPU-bound inference blocks async event loop  

**Solution:**
- Offloaded inference to ThreadPool / ProcessPool  
- Isolated I/O from compute  

**Impact:**
- Near-linear horizontal scaling (**R² ≈ 0.98**)  
- Efficient multi-worker utilization  

---

## 📊 System Performance

### 🔹 Throughput vs Latency

- **Peak Throughput:** 151.8 chunks/sec  
- **P99 Latency:** ~298 ms  
- Maintained SLA under sustained concurrent load  

---

### 🔹 Latency Breakdown

| Stage              | Time (ms) |
|-------------------|----------|
| Network I/O       | 40       |
| Queue Wait        | 60       |
| Inference         | 170      |
| Post-processing   | 28       |

---

### 🔹 Reliability Metrics

- Recovery Time: **~143 ms**
- Data Loss on Reconnect: **0%**
- Burst Stability: **P99 <400ms under 5× traffic spikes**

---

### 🔹 Scaling Efficiency

- Near-linear scaling (**R² ≈ 0.98**)  
- 1 worker → ~30 chunks/sec  
- 4 workers → ~115 chunks/sec  
- <5% inter-process overhead  

---

### 🔹 Resource Efficiency

- CPU Utilization Cap: **~85%**
- Memory per stream: **~50MB**
- Bandwidth: **~96 kbps per stream**

---

### 🔹 Saturation Behavior

- Stable up to ~150 chunks/sec  
- Beyond ~160 chunks/sec:
  - Controlled tail-drop (~4%)
  - No latency explosion
  - No system crashes  

---

## 🧠 ML Optimization

### Accuracy vs Latency Tradeoff

- Tested chunk sizes: 20ms / 100ms / 500ms  
- Final: **100ms VAD-filtered rolling window**

**Results:**
- **WER:** ~8–10%  
- **Latency:** <300ms  
- Robust under packet loss (~2%)  

---

### 💡 Key Insight

> Batching improves throughput but breaks conversational latency.  
> This system prioritizes **real-time responsiveness over raw throughput**.

---

## 🧪 Benchmarking

- Test File: `30_test.wav`
- Custom streaming benchmark framework
- Continuous real-time streaming validated
- No degradation under prolonged sessions

---

## 📡 Observability & Monitoring

- Fine-grained metrics:
  - `queue_wait_ms`
  - `inference_ms`
  - `end_to_end_latency`
- Designed for Prometheus / Grafana integration
- Supports alerting thresholds for autoscaling

---

## 🛠️ Tech Stack

| Layer | Technologies |
|------|-------------|
| **Frontend** | Next.js, React, TypeScript, Tailwind CSS |
| **Backend** | FastAPI, WebSockets, asyncio |
| **ML Models** | Faster-Whisper (INT8), LSTM |
| **Audio Processing** | WebRTC VAD, NumPy |

---

## 🎯 Key Learnings

- Designing **low-latency distributed systems**
- Handling **real-time streaming constraints**
- Tradeoffs between **latency, throughput, and accuracy**
- Importance of **backpressure, fairness, and load shedding**
- Observability as a **first-class production requirement**

---

## 📌 Future Improvements

- Redis / Kafka-based streaming pipeline
- GPU autoscaling for inference workloads
- Adaptive chunk sizing (dynamic latency tuning)
- Multilingual and code-switching optimization
- Kubernetes-based production deployment

---

## 🧩 Key Takeaways

- Real-time systems are **latency-bound, not throughput-bound**
- Backpressure is **mandatory**, not optional
- Controlled failure > unpredictable collapse
- Streaming ML systems require **systems + ML co-design**

---
