import os
import av
import torch
import numpy as np
import gc
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def process_video(video_path, is_violence=True):
    """Process a single video and generate caption"""
    try:
        # Set environment variable to avoid fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            device_id = 0
        else:
            device_id = "cpu"
        
        print(f"Testing video: {video_path}")
        print(f"Category: {'Violence' if is_violence else 'NonViolence'}")
        print(f"Using device: {'CUDA:0' if device_id == 0 else 'CPU'}")
        
        # Load model exactly as in the batch script
        model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
        
        print("Loading model...")
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(device_id)
        
        processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        
        # Define conversation based on category
        if is_violence:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This image contains violent or sensitive behavior. Please describe what you see in detail, focusing on any signs of violence, abnormal, dangerous or threatening actions."},
                        {"type": "video"},
                    ],
                },
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This image contains normal, peaceful activity. Please describe what you see in detail, focusing on the everyday, normal actions being performed."},
                        {"type": "video"},
                    ],
                },
            ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Open video file
        container = av.open(video_path)
        
        # Get total frames
        total_frames = container.streams.video[0].frames
        print(f"Total frames: {total_frames}")
        
        # Sample uniformly 8 frames from the video
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        
        # Read frames
        clip = read_video_pyav(container, indices)
        print(f"Extracted {len(clip)} frames")
        
        # Process inputs
        inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(device_id)
        
        # Generate caption
        print("Generating caption...")
        try:
            with torch.no_grad():
                output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
            caption = processor.decode(output[0][2:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            # Free memory
            torch.cuda.empty_cache()
            gc.collect()
            print("Out of memory! Trying with reduced frames...")
            
            # Try with reduced frames
            if len(clip) > 4:
                reduced_clip = clip[::2]  # Take every other frame
                inputs_video = processor(text=prompt, videos=reduced_clip, padding=True, return_tensors="pt").to(device_id)
                
                with torch.no_grad():
                    output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
                caption = processor.decode(output[0][2:], skip_special_tokens=True)
            else:
                return "Error: Out of memory and couldn't reduce frames further"
        
        # Free memory
        del inputs_video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("\nGenerated Caption:")
        print("="*80)
        print(caption)
        print("="*80)
        
        return caption
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return str(e)

def main():
    # Hardcoded parameters for easier testing on Kaggle
    # Để chạy trên local, bạn có thể thay đổi thành input như bên dưới
    # video_path = input("Enter path to a test video: ")
    # is_violence = input("Enter video category (Violence/NonViolence): ").lower() == "violence"
    
    # Hardcoded example - Thay đổi đường dẫn này để phù hợp với setup của bạn
    video_path = input("Enter path to a test video: ")
    
    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
    else:
        category = input("Enter video category (Violence/NonViolence): ")
        is_violence = category.lower() == "violence"
        process_video(video_path, is_violence)
    
    # Nếu muốn cố định đường dẫn (ví dụ trên Kaggle), bỏ comment dòng dưới và comment các dòng trên
    # process_video("/kaggle/input/violence-nonviolence-dataset/Violence/001/video.mp4", True)

if __name__ == "__main__":
    main() 