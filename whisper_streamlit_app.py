import streamlit as st
import whisper

# Initialize the model to None and load it on demand
model = None

def load_model(model_size='tiny'):
    global model
    model = whisper.load_model(model_size)

def transcribe_audio(audio_file):
    global model
    if not model:
        load_model()  # Load the model if it hasn't been loaded already
    return model.transcribe(audio_file)

def main():
    st.title('Audio Transcription with OpenAI Whisper')

    # Audio file uploader
    audio_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'ogg', 'webm'])

    if audio_file is not None:
        # Display an audio player widget
        st.audio(audio_file, format='audio/webm')  # Use the appropriate audio format

        # Confirm before transcribing
        if st.button('Transcribe Audio'):
            try:
                # Show a message while the transcription is being processed
                with st.spinner('Transcribing...'):
                    # Save the uploaded file to disk for Whisper processing
                    audio_file_path = audio_file.name
                    with open(audio_file_path, "wb") as f:
                        f.write(audio_file.getvalue())
                    
                    # Transcribe the audio
                    result = transcribe_audio(audio_file_path)
                    
                    # Display the transcription with timestamps
                    for segment in result["segments"]:
                        st.write(f"{segment['start']} - {segment['end']}: {segment['text']}")
            except Exception as e:
                # If an error occurs, display the error message
                st.error(f'An error occurred during transcription: {e}')

if __name__ == "__main__":
    main()