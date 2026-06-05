from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)




vectordb_file_path = "faiss_index"

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

    vectordb.save_local(vectordb_file_path)

    print("FAISS index created successfully!")





def get_qa_chain():

    vectordb = FAISS.load_local(
        vectordb_file_path,
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





def update_vector_db():

    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    new_data_folder = "../NewData"
    processed_file = "../processed_files.txt"

    # Read processed files
    if os.path.exists(processed_file):
        with open(processed_file, "r") as f:
            processed = set(line.strip() for line in f)
    else:
        processed = set()

    # Check all CSV files
    for file in os.listdir(new_data_folder):

        if not file.endswith(".csv"):
            continue

        if file in processed:
            continue

        file_path = os.path.join(new_data_folder, file)

        print(f"Processing {file}")

        loader = CSVLoader(
            file_path=file_path,
            source_column="prompt"
        )

        docs = loader.load()

        vectordb.add_documents(docs)

        with open(processed_file, "a") as f:
            f.write(file + "\n")

    vectordb.save_local(vectordb_file_path)

    print("Vector database updated successfully!")



if __name__ == "__main__":

    chain = get_qa_chain()

    response = chain.invoke(
        {"query": "Who directed Oppenheimer?"}
    )

    print(response["result"])

