# query_bot.py

import os, logging, warnings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Suppress unwanted warnings/logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)

# 1. Load FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore_index", embeddings, allow_dangerous_deserialization=True)


# 2. Load Llama2 model
llm = LlamaCpp(
    model_path="C:\\Users\\parth\\Desktop\\llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=4096,
    verbose=False
)

# 3. Prompt template
prompt = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question

Chat History: {chat_history}
Follow up Input: {question}
Standalone question:
""")

# 4. Create RAG chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    condense_question_prompt=prompt,
    return_source_documents=True,
    verbose=False
)

# 5. Ask a question
'''chat_history = []
query = "What are single-agent and multi-agent architectures? Explain in 3-4 lines."
result = qa.invoke({"question": query, "chat_history": chat_history})

# 6. Show the answer
print("\nAnswer:\n", result['answer'],"\n\n\n\n")'''

#6. Chat Loop
chat_history = []

print("Ask me anything from the document (type 'exit' to quit):\n")

while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    result = qa.invoke({
        "question": query,
        "chat_history": chat_history
    })

    answer = result["answer"]
    print(f"Bot: {answer}\n\n")

    # Update chat history with this Q&A pair
    chat_history.append((query, answer))

