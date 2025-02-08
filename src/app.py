from google.colab import drive
drive.mount("/content/drive")

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "BioMistral/BioMistral-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(f"MedicalChatbot/tokenizer/{model_name}") # Fixed f-string formatting

# Use AutoModelForCausalLM instead of AutoModelForMaskedLM
model = AutoModelForCausalLM.from_pretrained(model_name)  
model.save_pretrained(f"MedicalChatbot/model/{model_name}") # Fixed f-string formatting

#Install 
!pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf

#Importing Libraries
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import sentence_transformer
from langchain.vectorstores import chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain

#Import the document
loader = PyPDFDirectoryLoader("/content/MedicalChatbot/healthyheart.pdf")
docs = loader.load()
lens(docs)

#Chunking
text_splitter =  RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
chunk=text_splitter.split_documents(docs)
lens(chunk)

#Embeddings Creations
import os
os.environ["HuggingFaceHub_API_KEY"] = "TokenKey"
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

#vector store creation
vectorstore = Chorma.from_documents(chunks,embeddings)

query = "Who is at risk of heart disease"
search_result = vectorstore.similarity_search(query)

retriever = vectorstore.as_retriever(search_kwargs={'k':5})
retriever.get_relevant_documents(query)

#LLM Model Loading
llm = LlamaCpp{
model_path ="/content/MedicalChatbot/BioMistral/BioMistral-7B"
temperature = 0.2,
max_token = 2048,
top_p =1}

#use LLM and retriever and query to generate final response
template = """
<|context>
You are an Medical Assistant that follows the instructions and generate the accurate response based on the query and the context provided.
please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

from langchain.schema.runnable import RunnablePassThrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(template)

rag_chain = 
{ "context" : retriever, "query" : RunnablePassThrough()
 | prompt
 | llm
 | StrOutputParser()
}

response = rag_chain.invoke(query)
response

import sys
while True:
  user_input = input(f"Input query: ")
  if user_input =="exit":
    print("Existing....")
    sys.exit()
  if user_input="":
    continue
  result = raf_chain.invoker(user_input)
  print("Answer:",result)





