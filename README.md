# 🐻 Michael-DSPA: AI Peer Advisor

**Michael-DSPA** is a production-ready, RAG-powered (Retrieval-Augmented Generation) assistant designed to provide accurate academic advising for Data Science students at UC Berkeley. It leverages the "Michael" Peer Advising Archive to bridge the gap between official documentation and student-led wisdom.

---

## 🚀 System Architecture

This project follows a **Decoupled Cloud Architecture**, ensuring high availability and zero-cost maintenance on free-tier infrastructure.

| Layer | Technology | Hosting | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend** | Next.js 16 (React) | **Vercel** | Mobile-first UI with responsive "Hamburger" navigation. |
| **Backend** | FastAPI (Python) | **Render** | High-performance API with Gunicorn/Uvicorn workers. |
| **Vector DB** | Pinecone | **AWS Cloud** | Long-term "Memory" for 450+ indexed advising chunks. |
| **LLM Engine** | OpenAI GPT-4o | **API** | Advanced reasoning and context-aware response generation. |



---

## 🛠️ Technical Highlights

### 🧠 Advanced RAG Optimization
* **Granular Chunking:** Data is split into 500-character units with 100-character overlap to ensure high retrieval precision for specific entities like student clubs.
* **High-Recall Retrieval:** The system queries the top 15 most relevant vectors from Pinecone to provide comprehensive context to the LLM.
* **Batched Ingestion:** Optimized the data pipeline to handle Pinecone's 2MB payload limits via batched upserts (batch_size=50).

### 🛡️ The "Legal Fortress" (Security)
* **IP-Based Rate Limiting:** Implemented `SlowAPI` to restrict users to 10 requests per minute, protecting OpenAI credits without requiring user logins.
* **Proxy-Aware Security:** Configured to track `X-Forwarded-For` headers, ensuring rate limits apply to actual client IPs when deployed behind Render's load balancer.
* **CORS Protection:** Restricted backend access strictly to the Vercel production domain and local development environments.

### 📱 Mobile-First UX
* **Responsive Design:** Custom Tailwind CSS drawer for mobile navigation and touch-friendly targets (44x44px).
* **Input Optimization:** Standardized 16px font sizes to prevent intrusive iOS Safari "auto-zoom" bugs.
* **Suggested Actions:** Integrated static "Quick Action" buttons to guide students toward high-value advising paths.

---

## 🏗️ Local Development

1. **Backend:**
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
2. **Frontend:**
   Bash
   cd frontend
   npm install
   npm run dev
