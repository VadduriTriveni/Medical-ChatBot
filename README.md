# Medical-ChatBot
Medical Chatbot - with BioMistral Open Source LLM

# Data
HealthyHeart

# Frame works
Langchain -Pipeline
Llama -LLM Model
Sentence -Transformer -Embedding Model
Chroma -Vector Store

# Models
LLM - Biomistral-7B
Embedding - PubMedBert_Base_Embedding

# Process
Two Parts
# Part 1: Indexing
1.Load the document and parse the text
2.Divide Text into Chuncks - Chuncking
3.Create embedding vector for each chunck
4.Store chunks and embeddings to vector store

# Part 2: Querying
1.Load LLM model
2.Build application chain end to end
3.Query the chatbot
   3.1.Pass  query to retriever
	 3.2.Retrieves relevant docs from vector store(KNN)
	 3.3.Pass both query and docs to LLM
	 3.4.Generate the response







