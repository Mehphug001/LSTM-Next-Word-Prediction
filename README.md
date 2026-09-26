# 🧠 Next Word Prediction using LSTM & Streamlit

An end-to-end Deep Learning Natural Language Processing (NLP) application that predicts the next likely word in a sentence using a Long Short-Term Memory (LSTM) Recurrent Neural Network, deployed with an interactive web UI via Streamlit.

---

## 🚀 Live Demo
🔗 **Live App:** [Deployed on Streamlit Community Cloud](https://share.streamlit.io) *(replace with your app URL)*  
📂 **GitHub Repository:** [Mehphug001/LSTM-Next-Word-Prediction](https://github.com/Mehphug001/LSTM-Next-Word-Prediction)

---

## 📌 Project Overview
Language modeling and next-word prediction are fundamental to modern search engines, autocomplete features, and generative AI. In this project:
- A dataset of **3,000+ literary quotes** was preprocessed and tokenized.
- Sub-sequences were created using a rolling context window to model temporal word dependencies.
- An **LSTM network** with embedding and dropout layers was trained using TensorFlow/Keras.
- An interactive **Streamlit** web application was built and deployed to provide real-time word inference.

---

## 🏗️ Architecture & Model Pipeline

```
Raw Quotes Data
      │
      ▼
Text Preprocessing (Lowercasing, Punctuation Cleaning)
      │
      ▼
Tokenization & Vocabulary Mapping (8,979 tokens)
      │
      ▼
Sliding Context Sequencing (Context Window = 50)
      │
      ▼
Embedding Layer (Input: 8979, Output: 64)
      │
      ▼
LSTM Layer (128 Units)
      │
      ▼
Dropout (0.2 Regularization)
      │
      ▼
Dense Softmax Layer (Output: 8979 Word Probabilities)
      │
      ▼
Interactive Streamlit UI
```

---

## 🛠️ Tech Stack & Tools
- **Deep Learning Framework:** TensorFlow / Keras
- **Model Architecture:** Embedding + LSTM + Dropout + Dense
- **Deployment & UI:** Streamlit
- **Language & Libraries:** Python, NumPy, Pandas, Pickle

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Mehphug001/LSTM-Next-Word-Prediction.git
cd LSTM-Next-Word-Prediction
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```

---

## 🌟 Example Queries
Try typing any of these seed phrases to test predictions:
- `the meaning of life` ➔ **is**
- `what you` ➔ **are**
- `the world` ➔ **is**
- `imperfection is` ➔ **beauty**
- `without changing our` ➔ **thinking**

---

## 👤 Author
- **Sk Mehphug Rahaman**
- **GitHub:** [@Mehphug001](https://github.com/Mehphug001)
