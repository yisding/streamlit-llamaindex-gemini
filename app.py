import os
from io import BytesIO

import openai
import requests
import streamlit as st
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.generic_utils import load_image_urls

os.environ["GOOGLE_API_KEY"] = st.secrets["google_key"]

image_urls = [
    "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
]

st.set_page_config(
    page_title="Chat with images, powered by LlamaIndex, Streamlit, and Gemini",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
openai.api_key = st.secrets.openai_key
st.title("Chat with images, powered by LlamaIndex, Streamlit, and Gemini")

if "messages" not in st.session_state.keys():
    # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about this image!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading images"):
        image_documents = load_image_urls(image_urls)

        return image_documents


image_documents = load_data()
gemini_pro = GeminiMultiModal(model="models/gemini-pro")


img_response = requests.get(image_urls[0])
st.image(BytesIO(img_response.content))

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if prompt is not None:
                response = gemini_pro.complete(
                    prompt=prompt, image_documents=image_documents
                )
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
