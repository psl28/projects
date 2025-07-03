# embed_documents.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the PDF
pdf_path = "C:\\Users\\parth\\Desktop\\python\\Custom_rb_Ollama\\wtmini.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Save FAISS vectorstore
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore_index")

print("✅ Vector store saved successfully to ./vectorstore_index")
