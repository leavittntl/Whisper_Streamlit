import streamlit as st
import whisper

# Load Whisper model - consider doing this outside of the function to avoid reloading for each user
model = whisper.load_model("base")

def transcribe_audio(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

st.title("Whisper Audio Transcription App")

uploaded_file = st.file_uploader("Choose an audio file...")

if uploaded_file is not None:
    with st.spinner("Transcribing..."):
        # Save uploaded file to a temporary location
        audio_file_name = f'/mnt/data/{uploaded_file.name}'
        with open(audio_file_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Transcribe the audio
        transcription = transcribe_audio(audio_file_name)
        
        # Display the results
        st.subheader("Transcription Result")
        st.write(transcription)
else:
    st.write("Upload an audio file to get started.")