import streamlit as st
import whisper
import soundfile as sf
from io import BytesIO

# Load Whisper model
@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None, "builtins.weakref": lambda _: None})
def load_model():
    return whisper.load_model("base")

model = load_model()

def transcribe_audio(audio_data):
    # Convert BytesIO object to NumPy array expected by Whisper
    with sf.SoundFile(audio_data) as audio:
        samples = audio.read(dtype="float32")
    result = model.transcribe(samples)
    return result["text"]

st.title("Whisper Audio Transcription App")

uploaded_file = st.file_uploader("Choose an audio file...", type=['mp3', 'wav', 'flac', 'ogg', 'mp4'])

if uploaded_file is not None:
    bytes_io = uploaded_file.getvalue()
    buffer = BytesIO(bytes_io)
    
    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(buffer)
        
        # Display the results
        st.subheader("Transcription Result")
        st.write(transcription)
else:
    st.write("Upload an audio file to get started.")