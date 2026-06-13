import streamlit as st

from medical_helper import detect_medical_entities

from langchain_helper import (
    get_qa_chain,
    get_medical_qa_chain,
    get_arxiv_qa_chain,
    create_vector_db
)

from image_helper import analyze_image


st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Multi-Modal AI Assistant")

mode = st.selectbox(
    "Select Assistant Mode",
    [
        "Customer Service",
        "Medical Assistant",
        "ArXiv Research Assistant"
    ]
)

st.divider()

# ---------------------------
# Session State
# ---------------------------

if "image_context" not in st.session_state:
    st.session_state.image_context = ""

# ---------------------------
# Image Upload
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image_result = analyze_image(uploaded_file)

    st.session_state.image_context = image_result

    st.subheader("📷 Image Analysis")
    st.write(image_result)

st.divider()

# ---------------------------
# Customer Knowledge Base
# ---------------------------

if mode == "Customer Service":

    btn = st.button(
        "Create Customer Knowledge Base"
    )

    if btn:
        create_vector_db()

# ---------------------------
# Question Input
# ---------------------------

question = st.text_input(
    "Ask a Question"
)

# ---------------------------
# Processing
# ---------------------------

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

    # ---------------------------
    # Medical Entity Recognition
    # ---------------------------

    if mode == "Medical Assistant":

        entities = detect_medical_entities(
            question
        )

        if entities:

            st.subheader(
                "🩺 Detected Medical Entities"
            )

            for entity in entities:
                st.write(entity)

    # ---------------------------
    # Select Chain
    # ---------------------------

    if mode == "Customer Service":

        chain = get_qa_chain()

    elif mode == "Medical Assistant":

        chain = get_medical_qa_chain()

    elif mode == "ArXiv Research Assistant":

        chain = get_arxiv_qa_chain()
    # ---------------------------
    # Generate Response
    # ---------------------------

    try:

        with st.spinner("Generating response..."):

            response = chain.invoke(
                {"query": final_question}
            )

        st.subheader("💡 Answer")
        st.write(response["result"])

    except Exception as e:

        st.error(str(e))