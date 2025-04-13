# Mount Google Drive
from google.colab import drive
drive.mount("/content/drive")

# Install necessary packages
!pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf

# Import necessary libraries
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassThrough
from langchain.schema.output_parser import StrOutputParser

# Step 1: Load and save tokenizer and model
model_name = "BioMistral/BioMistral-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(f"/content/MedicalChatbot/tokenizer/{model_name}")

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(f"/content/MedicalChatbot/model/{model_name}")

# Step 2: Load PDF document
pdf_path = "/content/MedicalChatbot/healthyheart.pdf"
loader = PyPDFDirectoryLoader(pdf_path)
docs = loader.load()
print("✅ Documents loaded:", len(docs))

# Step 3: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print("✅ Chunks created:", len(chunks))

# Step 4: Create sentence-transformer embeddings
os.environ["HuggingFaceHub_API_KEY"] = "TokenKey"  # Replace with your Hugging Face key
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Step 5: Create Chroma vectorstore
vectorstore = Chroma.from_documents(chunks, embeddings)
print("✅ Vector store created")

# Step 6: Run a test query
query = "Who is at risk of heart disease"
search_result = vectorstore.similarity_search(query)
print("✅ Similarity search results found:", len(search_result))

# Step 7: Create retriever from vectorstore
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
retrieved_docs = retriever.get_relevant_documents(query)
print("✅ Retrieved relevant documents:", len(retrieved_docs))

# Step 8: Load the LLaMA model using LlamaCpp
llm = LlamaCpp(
    model_path="/content/MedicalChatbot/model/BioMistral/BioMistral-7B",
    temperature=0.2,
    max_tokens=2048,
    top_p=1
)
print("✅ LLM loaded")

# Step 9: Define prompt template
template = """
<|context>
You are a Medical Assistant that follows the instructions and generates accurate responses based on the query and the context provided.
Please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)

# Step 10: Create the Retrieval-Augmented Generation (RAG) chain
rag_chain = (
    {"context": retriever, "query": RunnablePassThrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 11: Test initial response
response = rag_chain.invoke(query)
print("🧠 Initial Test Response:\n", response)

# Step 12: Start interactive loop
print("\n💬 Medical Assistant Chatbot is ready! Type 'exit' to quit.\n")

while True:
    user_input = input("Input query: ").strip()
    if user_input.lower() == "exit":
        print("👋 Exiting...")
        sys.exit()
    if user_input == "":
        continue
    result = rag_chain.invoke(user_input)
    print("🩺 Answer:", result)
