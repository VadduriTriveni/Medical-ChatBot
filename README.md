## 🩺 Medical Chatbot using BioMistral-7B + LangChain

An intelligent, Retrieval-Augmented Generation (RAG) chatbot designed to answer medical queries using the [BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B) language model and LangChain. This chatbot processes medical documents (PDFs) and responds to user queries based on the contextual knowledge retrieved from them.

---

### 🚀 Features

- ✅ Loads **BioMistral-7B**, a domain-specific LLM trained on biomedical literature
- ✅ Uses **LangChain's RAG pipeline** for intelligent query responses
- ✅ Handles PDF medical documents and splits them into context-rich chunks
- ✅ Employs **ChromaDB** for semantic vector storage
- ✅ Uses **SentenceTransformer** for embedding medical text
- ✅ Interactive CLI chatbot in Colab or terminal
- ✅ Powered by `llama-cpp-python` for local LLM inference

---

### 🛠️ Installation & Setup

#### 1. Clone this Repository

```bash
git clone https://github.com/yourusername/MedicalChatbot.git
cd MedicalChatbot
```

#### 2. Open in Google Colab or Jupyter

This code is designed to run inside **Google Colab**. Ensure the document `healthyheart.pdf` is in the correct path.

#### 3. Required Dependencies

Install the required libraries:

```bash
pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf
```

---

### 🧠 How It Works

1. **Mount Google Drive** to access PDF documents and save model artifacts.
2. **Load BioMistral-7B** from Hugging Face.
3. **Split medical PDFs** into smaller text chunks.
4. **Embed those chunks** using `SentenceTransformerEmbeddings`.
5. **Store them in ChromaDB** for efficient vector-based retrieval.
6. **User query is matched** with the most relevant chunks.
7. **BioMistral generates responses** based on retrieved context.

---

### 💬 Run the Chatbot

Once everything is set up, just run the final loop:

```python
while True:
    user_input = input("Input query: ")
    ...
```

Example:

```
Input query: What are the symptoms of heart disease?
🩺 Answer: Common symptoms include chest pain, shortness of breath, fatigue, and palpitations.
```

Type `exit` to quit the assistant.

---

### 🔐 API Keys

You must provide a Hugging Face API key to use certain models:

```python
os.environ["HuggingFaceHub_API_KEY"] = "your_huggingface_api_key"
```

### 📌 Requirements

- Python 3.8+
- Google Colab or Jupyter
- Hugging Face account (for model access)
- PDF file of medical content (like `healthyheart.pdf`)

---

### 🤖 Model Credits

- **[BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B)**: Biomedical-focused LLM based on Mistral architecture
- **SentenceTransformer**: `NeuML/pubmedbert-base-embeddings`

### 🙌 Acknowledgements

Thanks to:
- [Hugging Face](https://huggingface.co/)
- [LangChain](https://www.langchain.com/)
- [Mistral AI](https://mistral.ai/)
- [ChromaDB](https://www.trychroma.com/)

---
