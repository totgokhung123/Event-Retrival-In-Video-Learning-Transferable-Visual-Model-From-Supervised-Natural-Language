import os
import av
import json
import torch
import argparse
import numpy as np
import gc
from tqdm import tqdm
from huggingface_hub import hf_hub_download
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

def process_video(video_path, model, processor, is_violence, device_id=0):
    try:
        # Open video file
        container = av.open(video_path)
        
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
        
        # sample uniformly 8 frames from the video
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)
        
        inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(device_id)
        
        try:
            with torch.no_grad():
                output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
            caption = processor.decode(output[0][2:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            # Free memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Try with reduced frames
            if len(clip) > 4:
                reduced_clip = clip[::2]  # Take every other frame
                inputs_video = processor(text=prompt, videos=reduced_clip, padding=True, return_tensors="pt").to(device_id)
                
                with torch.no_grad():
                    output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
                caption = processor.decode(output[0][2:], skip_special_tokens=True)
            else:
                return f"Error: Out of memory and couldn't reduce frames further for {video_path}"
                
        # Free memory
        del inputs_video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
            
        return caption
        
    except Exception as e:
        return f"Error processing {video_path}: {str(e)}"

def main():
    # Giá trị cứng cho các tham số thay vì dùng argparse
    # Cập nhật đường dẫn phù hợp với cấu trúc thư mục trên Kaggle
    dataset_path = "/kaggle/input/violence-nonviolence-dataset"  # Đường dẫn đến dataset trên Kaggle
    output_file = "/kaggle/working/llava_captions.json"  # Đường dẫn lưu kết quả trên Kaggle
    max_videos = 1000  # Số lượng video tối đa mỗi danh mục
    save_interval = 50  # Lưu kết quả trung gian sau mỗi bao nhiêu video
    
    # Class giả lập để giữ cấu trúc code nhất quán
    class Args:
        def __init__(self):
            self.dataset_path = dataset_path
            self.output_file = output_file
            self.max_videos = max_videos
            self.save_interval = save_interval
    
    args = Args()
    
    print(f"Using fixed parameters:")
    print(f"- Dataset path: {args.dataset_path}")
    print(f"- Output file: {args.output_file}")
    print(f"- Max videos per category: {args.max_videos}")
    print(f"- Save interval: {args.save_interval}")
    
    # Set environment variable to avoid fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device_id = 0
    else:
        device_id = "cpu"
        print("CUDA not available, using CPU")
    
    # Load model exactly as in the example code
    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    
    print("Loading model...")
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(device_id)
    
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    
    results = {}
    processed_count = 0
    
    # Process Violence videos
    violence_path = os.path.join(args.dataset_path, "Violence")
    if os.path.exists(violence_path):
        print(f"\nProcessing Violence videos from {violence_path}")
        # Changed this part to process video files directly in the Violence folder
        video_files = [f for f in sorted(os.listdir(violence_path)) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))][:args.max_videos]
        
        for video_file in tqdm(video_files):
            video_path = os.path.join(violence_path, video_file)
            
            if not os.path.isfile(video_path):
                continue
                
            # Process video
            caption = process_video(video_path, model, processor, is_violence=True, device_id=device_id)
            
            # Store result
            results[video_path] = {
                "category": "Violence",
                "caption": caption
            }
            
            processed_count += 1
            
            # Print sample results
            if processed_count <= 2:
                print(f"\nSample caption for Violence video {video_path}:")
                print(f"{caption}\n{'='*80}")
                
            # Save intermediate results
            if processed_count % args.save_interval == 0:
                interim_file = f"{os.path.splitext(args.output_file)[0]}_interim_{processed_count}.json"
                with open(interim_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Saved interim results to {interim_file}")
                
            # Clear GPU memory periodically
            if processed_count % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    # Process NonViolence videos
    nonviolence_path = os.path.join(args.dataset_path, "NonViolence")
    if os.path.exists(nonviolence_path):
        print(f"\nProcessing NonViolence videos from {nonviolence_path}")
        # Changed this part to process video files directly in the NonViolence folder
        video_files = [f for f in sorted(os.listdir(nonviolence_path)) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))][:args.max_videos]
        
        for video_file in tqdm(video_files):
            video_path = os.path.join(nonviolence_path, video_file)
            
            if not os.path.isfile(video_path):
                continue
                
            # Process video
            caption = process_video(video_path, model, processor, is_violence=False, device_id=device_id)
            
            # Store result
            results[video_path] = {
                "category": "NonViolence",
                "caption": caption
            }
            
            processed_count += 1
            
            # Print sample results for first few videos
            if processed_count % 1000 <= 2:
                print(f"\nSample caption for NonViolence video {video_path}:")
                print(f"{caption}\n{'='*80}")
                
            # Save intermediate results
            if processed_count % args.save_interval == 0:
                interim_file = f"{os.path.splitext(args.output_file)[0]}_interim_{processed_count}.json"
                with open(interim_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Saved interim results to {interim_file}")
                
            # Clear GPU memory periodically
            if processed_count % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    # Save final results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessed {len(results)} videos. Results saved to {args.output_file}")
    
    # Print summary
    violence_count = sum(1 for k, v in results.items() if v["category"] == "Violence")
    nonviolence_count = sum(1 for k, v in results.items() if v["category"] == "NonViolence")
    print(f"Violence videos captioned: {violence_count}")
    print(f"NonViolence videos captioned: {nonviolence_count}")

if __name__ == "__main__":
    main() 