from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# Gemini
# --------------------------------------------------

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

customer_vectordb_path = "faiss_index"

medical_vectordb_path = "faiss_index/medical_index"

# --------------------------------------------------
# Customer Service Vector DB
# --------------------------------------------------

def create_vector_db():

    loader = CSVLoader(
        file_path="../dataset/dataset.csv",
        source_column="prompt"
    )

    data = loader.load()

    vectordb = FAISS.from_documents(
        documents=data,
        embedding=embeddings
    )

    vectordb.save_local(customer_vectordb_path)

    print("Customer FAISS index created successfully!")

# --------------------------------------------------
# Medical Vector DB
# --------------------------------------------------

def create_medical_vector_db():

    loader = CSVLoader(
        file_path="../dataset/Medical/medical_dataset.csv",
        source_column="prompt",
        encoding="utf-8"
    )

    data = loader.load()

    print(f"Medical documents loaded: {len(data)}")

    vectordb = FAISS.from_documents(
        documents=data,
        embedding=embeddings
    )

    vectordb.save_local(medical_vectordb_path)

    print("Medical FAISS index created successfully!")

# --------------------------------------------------
# Customer QA Chain
# --------------------------------------------------

def get_qa_chain():

    vectordb = FAISS.load_local(
        customer_vectordb_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt_template = """
    Use only the provided context to answer.

    If the answer is not present in the context,
    respond with:
    I don't know.

    Context:
    {context}

    Question:
    {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

# --------------------------------------------------
# Medical QA Chain
# --------------------------------------------------

def get_medical_qa_chain():

    vectordb = FAISS.load_local(
        medical_vectordb_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt_template = """
    You are a medical assistant.

    Use only the provided medical context.

    If the answer is not present in the context,
    respond with:
    I don't know.

    Context:
    {context}

    Question:
    {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

# --------------------------------------------------
# Dynamic Update System (Task 1)
# --------------------------------------------------

def update_vector_db():

    vectordb = FAISS.load_local(
        customer_vectordb_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    new_data_folder = "../NewData"
    processed_file = "../processed_files.txt"

    if os.path.exists(processed_file):

        with open(processed_file, "r") as f:
            processed = set(
                line.strip()
                for line in f
            )

    else:
        processed = set()

    for file in os.listdir(new_data_folder):

        if not file.endswith(".csv"):
            continue

        if file in processed:
            continue

        file_path = os.path.join(
            new_data_folder,
            file
        )

        print(f"Processing {file}")

        loader = CSVLoader(
            file_path=file_path,
            source_column="prompt"
        )

        docs = loader.load()

        vectordb.add_documents(docs)

        with open(processed_file, "a") as f:
            f.write(file + "\n")

    vectordb.save_local(customer_vectordb_path)

    print("Vector database updated successfully!")

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    create_medical_vector_db()