import streamlit as st
import whisper
from io import BytesIO

# Load Whisper model - consider doing this outside of the function to avoid reloading for each user
model = whisper.load_model("base")

@st.cache(allow_output_mutation=True)
def load_model():
    return whisper.load_model("base")

model = load_model()

def transcribe_audio(audio_data):
    audio_buffer = BytesIO(audio_data)
    result = model.transcribe(audio_buffer)
    return result["text"]

st.title("Whisper Audio Transcription App")

uploaded_file = st.file_uploader("Choose an audio file...")

if uploaded_file is not None:
    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(uploaded_file.getvalue())
        
        # Display the results
        st.subheader("Transcription Result")
        st.write(transcription)
else:
    st.write("Upload an audio file to get started.")