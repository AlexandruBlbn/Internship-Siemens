import whisper
import torch

def transcribe_audio(audio_file_path):
    """
    Transcribe audio using OpenAI's Whisper small model (CPU version).
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        dict: Transcription result
    """
    # Force CPU usage
    device = torch.device("cpu")
    
    # Load the small model on CPU
    print("Loading Whisper small model on CPU...")
    model = whisper.load_model("small", device=device)
    
    print(f"Transcribing: {audio_file_path}")
    result = model.transcribe(audio_file_path)
    return result

if __name__ == "__main__":
    # Replace with your audio file path
    audio_path = "your_audio_file.mp3"  # Update this to your audio file path
    
    try:
        result = transcribe_audio(audio_path)
        print("\nTranscription:")
        print(result["text"])
    except Exception as e:
        print(f"Error during transcription: {e}")