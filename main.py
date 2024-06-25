import subprocess
import sys

def install_libraries():
    print("Step 1 - Installing required libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytube", "--quiet"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "git+https://github.com/huggingface/transformers.git", "accelerate", "datasets[audio]", "--quiet"])

def ilmstore_youtube_transcribe():
    install_libraries()
    print("Alhamdulillah, required code installed successfully. - Step 1 Complete") 

    print("Step 2 - Importing code for use now.") 
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from datasets import load_dataset

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print ("I am processing it with device type :",device) 
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]
    return pipe

def download_video (videoURL):
    import pytube
    print ("Downloading YouTube Video")
    data = pytube.YouTube(videoURL)
    # Convert to audio file
    audio = data.streams.get_audio_only()
    VideoPath = audio.download()
    print ("Download Complete YouTube Video")
    return VideoPath

def ilmstore_transcriber (videoPath, pipe):
    print ("Starting transcribing now")
    result = pipe(videoPath)
    print ("Transcription Done")
    return result
    


