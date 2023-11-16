import streamlit as st
import whisper
import os

# Initialize the model to None and load it on demand
model = None

def load_model(model_size='tiny'):
    global model
    model = whisper.load_model(model_size)

def transcribe_audio(audio_file_path):
    global model
    if not model:
        load_model()  # Load the model if it hasn't been loaded already
    return model.transcribe(audio_file_path)

def main():
    st.title('Audio Transcription with OpenAI Whisper')

    # Audio file uploader
    audio_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'ogg', 'webm'])

    if audio_file is not None:
        # Display an audio player widget
        st.audio(audio_file, format='audio/webm')

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
                    
                    # Write the transcription text to a string
                    transcription_text = ''
                    for segment in result["segments"]:
                        transcription_text += f"{segment['start']} - {segment['end']}: {segment['text']}\n"
                    
                    # Display the transcription with timestamps
                    st.text_area('Transcription', transcription_text, height=300)
                    
                    # Let the user download the transcription
                    st.download_button(
                        label="Download Transcription",
                        data=transcription_text,
                        file_name="transcription.txt",
                        mime="text/plain"
                    )
                    
                    # Clean up the audio file saved on disk
                    if os.path.exists(audio_file_path):
                        os.remove(audio_file_path)
                    
            except Exception as e:
                # Clean up the audio file saved on disk in case of an error
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                # If an error occurs, display the error message
                st.error(f'An error occurred during transcription: {e}')

if __name__ == "__main__":
    main()