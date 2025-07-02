![PYTHON](https://img.shields.io/badge/Python-3.8+-blue.svg)
![GITHUB_API](https://img.shields.io/badge/GitHub-API-lightgrey.svg)
![LSTM](https://img.shields.io/badge/ML-LSTM%252BAttention-orange.svg)
# ChatOra
AI-powered chat assistant for GitHub-related technical questions, built from scratch with custom neural network architecture.
# Features
•**Pure Python implementation** - No TensorFlow/PyTorch required

•**Context-aware responses** - LSTM with Attention mechanism

•**GitHub integration** - Analyze repositories and code

•**Technical Q&A** - 300+ pre-trained programming concepts

•**Web API** - FastAPI interface for easy integration
# Installation
1. Clone repository:
```bash
git clone https://github.com/ImHacker890-890/ChatOra.git
cd ChatOra
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Add your GitHub token (optional for code analysis):
```bash
echo "GITHUB_TOKEN=your_personal_access_token" > .env
```
# Usage
**Training the Model**
```bash
python train.py
```
**Running the web interface**
```bash
uvicorn app:app --reload
```
**Example API calls**
```bash
# Chat endpoint
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d '{"message":"how to resolve git merge conflict"}'

# GitHub analysis endpoint
curl -X GET "http://localhost:8000/analyze?repo=ImHacker890-890/ChatOra"
```
# Project Structure
```text
ChatOra/
├── core_lstm.py
├── core_attention.py
├── core_model.py
├── data_loader.py
├── github_api.py
├── train.py
├── app.py
├── requirements.txt
└── data/
    ├── dialogues.txt
```
# Customization
**Adding more training data**
Edit data/dialogues.txt using the format:
```txt
question|answer
how to clone a repo|Use git clone https://github.com/user/repo.git
```
**Modifying model architecture**
Adjust hyperparameters in core/model.py:
```py
class NeuralChat:
    def __init__(self, vocab_size=10000, hidden_size=256):
        # Network architecture
        self.lstm = LSTMLayer(vocab_size, hidden_size)
        self.attention = Attention(hidden_size)
```
# License
MIT License - Feel free to use and modify for your projects!
# Contributing
Pull requests welcome! For major changes, please open an issue first to discuss proposed changes.
