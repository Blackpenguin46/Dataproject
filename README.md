# Combined Dashboard: BPO, Data Science Assistant & Local Chatbot

This project is a multi-tool dashboard built with Streamlit, designed to help you:
- **Optimize business processes** (BPO Dashboard)
- **Analyze data and train simple ML models** (Data Science Assistant)
- **Chat with a local AI assistant** (Ollama-powered Chatbot)

## Features
- 📈 **BPO Dashboard**: Add, view, and analyze business processes and their optimization goals.
- 🧠 **Data Science Assistant**: Upload CSVs, select features, and train a linear regression model with performance metrics and plots.
- 🤖 **Chatbot**: Ask questions to a local LLM (Mistral via Ollama) for privacy and zero cost.

---

## Quick Start (Local)
1. **Install Python 3.12+**
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Install and run Ollama:**
   - Download from [https://ollama.com/download](https://ollama.com/download)
   - Start the model:
     ```sh
     ollama run mistral
     ```
4. **Run the app:**
   ```sh
   streamlit run test.py
   ```
5. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Quick Start (Docker Compose)
1. **Install Docker & Docker Compose**
2. **Build and run everything:**
   ```sh
   docker compose up --build
   ```
3. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Sharing & Collaboration
- Push this project to a GitHub repo:
  ```sh
  git init
  git add .
  git commit -m "Initial commit"
  git remote add origin <your-repo-url>
  git push -u origin main
  ```
- Share the repo link with your friend. They can clone and follow the setup instructions above.

## Requirements
- Python 3.12+
- Docker (for containerized setup)
- Ollama (for local LLM chatbot)

---

## License
MIT (or specify your preferred license)