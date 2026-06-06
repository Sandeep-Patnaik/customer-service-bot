import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
from image_helper import analyze_image


st.title(" CUSTOMER SERVICE CHATBOT 🤖")




uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if "image_context" not in st.session_state:
    st.session_state.image_context = ""



if uploaded_file:

    image_result = analyze_image(uploaded_file)

    st.session_state.image_context = image_result

    st.subheader("Image Analysis")
    st.write(image_result)







btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:

    final_question = question

    if st.session_state.image_context:

        final_question = f"""
        Image Context:
        {st.session_state.image_context}

        User Question:
        {question}

        Use the image context if relevant.
        """

    chain = get_qa_chain()

    response = chain.invoke(
        {"query": final_question}
    )

    st.header("Answer")
    st.write(response["result"])