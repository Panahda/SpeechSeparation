import numpy as np
import soundfile as sf
from pydub import AudioSegment
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from moviepy.editor import VideoFileClip, AudioFileClip
import torch
import io
import os

# Specific GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  

def speech_separation(input_mp4, output_mp4):
    sample_rate = 8000
    channels = 1
    audio = AudioSegment.from_file(input_mp4, format="mp4")
    audio = audio.set_frame_rate(sample_rate).set_channels(channels)
    
    # Select desinated temp.wav location
    temp_wav = 'temp.wav'
    audio.export(temp_wav, format="wav")
    print(f"""
    ************************************************************************************************\n
    Converted to temp.wav with {sample_rate} Hz and {channels} channel(s).
    ************************************************************************************************\n""")

    # Ensure PyTorch uses GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Initialize the pipeline with GPU support
    separation = pipeline(
        Tasks.speech_separation,
        model='damo/speech_mossformer2_separation_temporal_8k',
        device=device
    )
    
    # Perform the separation
    result = separation(temp_wav)

    # Save the output
    # Select desinated temp.wav location
    temp_wav = f'temp.wav'
    sf.write(temp_wav, np.frombuffer(result['output_pcm_list'][0], dtype=np.int16), 8000)
    
    print(f"""
    ************************************************************************************************\n
    Speech Isolation Completed. Saved separated speaker to temp.wav.
    ************************************************************************************************\n""")
    
    video = VideoFileClip(input_mp4)
    
    audio = AudioFileClip(temp_wav)
    new_video = video.set_audio(audio)
    new_video.write_videofile(output_mp4, codec='libx264', audio_codec='aac')
    
    # Clean up temporary file
    os.remove(temp_wav)

    print(f"""
    ************************************************************************************************\n
    Saved new video with replaced audio: {output_mp4}
    ************************************************************************************************\n""")

if __name__ == "__main__":
    input_mp4 = r"/home/zongjiawen/SpeechSep/input/noisy_speech_2.mp4"
    output_mp4 = r"/home/zongjiawen/SpeechSep/output/separated_speech_2.mp4"
    speech_separation(input_mp4, output_mp4)