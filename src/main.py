import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
from image_helper import analyze_image


st.title(" CUSTOMER SERVICE CHATBOT 🤖")




uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image_result = analyze_image(uploaded_file)

    st.subheader("Image Analysis")

    st.write(image_result)




btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain.invoke(
    {"query": question}
)

    st.header("Answer")
    st.write(response["result"])
