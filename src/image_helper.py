from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import os

from streamlit import image

load_dotenv()


genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")


def analyze_image(uploaded_file):

    image = Image.open(uploaded_file)

    prompt = """
    Analyze this image.

    Provide:
    1. What is visible.
    2. Important text if any.
    3. A short summary.
    """

    response = model.generate_content(
        [prompt, image]
    )

    return response.text