"""
Background Music Generation Tool - Generates contextual music for videos
"""
import json
import os
import tempfile
from typing import Dict, Any, Optional
from langchain.tools import tool
from pydantic import BaseModel, Field
import replicate
import time
from langchain_openai import ChatOpenAI


class GeneratedMusic(BaseModel):
    """Generated music information"""
    music_url: str = Field(description="URL of the generated music")
    prompt: str = Field(description="Prompt used for generation")
    duration: int = Field(description="Duration in seconds")
    model_used: str = Field(description="Model used for generation")
    generation_time: float = Field(description="Time taken to generate in seconds")
    success: bool = Field(description="Whether generation was successful", default=True)
    error_message: Optional[str] = Field(description="Error message if failed", default=None)


def generate_music_prompt_with_llm(video_theme: str, scene_descriptions: str = "") -> str:
    """Generate music prompt using LLM based on video theme and scenes"""
    
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        prompt_template = f"""Generate a detailed music prompt for background music that matches this video content:

Video Theme: {video_theme}
Scene Descriptions: {scene_descriptions}

Create a music prompt that describes:
- Musical style and genre
- Instruments to include
- Mood and atmosphere
- Energy level and tempo
- Cultural elements if relevant

Keep the prompt concise but descriptive. Focus on the musical elements, not the duration.

Example: "Uplifting Indian classical fusion with tabla, sitar, and modern orchestral elements. Festive celebratory mood with traditional instruments."

Music Prompt:"""

        response = llm.invoke(prompt_template)
        music_prompt = str(response.content).strip() if hasattr(response, 'content') else str(response).strip()
        
        print(f"DEBUG - LLM generated music prompt: {music_prompt}")
        return music_prompt
        
    except Exception as e:
        print(f"ERROR - Failed to generate music prompt with LLM: {e}")
        # Fallback to simple prompt
        return f"Cinematic background music suitable for {video_theme}"


def generate_music_with_replicate(prompt: str, duration: int) -> str:
    """Generate music using Meta MusicGen via Replicate"""
    try:
        print(f"DEBUG - Generating music with prompt: {prompt}")
        print(f"DEBUG - Duration: {duration} seconds")
        
        # MusicGen parameters
        input_params = {
            "top_k": 250,
            "top_p": 0,
            "prompt": prompt,
            "duration": duration,
            "temperature": 1,
            "continuation": False,
            "model_version": "stereo-large",
            "output_format": "mp3",
            "continuation_start": 0,
            "multi_band_diffusion": False,
            "normalization_strategy": "peak",
            "classifier_free_guidance": 3
        }
        
        print(f"DEBUG - MusicGen parameters: {input_params}")
        
        # Generate music
        output = replicate.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input=input_params
        )
        
        # Handle different output formats
        if isinstance(output, str):
            music_url = output
        elif isinstance(output, (list, tuple)) and output:
            music_url = str(output[0])
        else:
            music_url = str(output)
        
        print(f"DEBUG - Generated music URL: {music_url}")
        return music_url
        
    except Exception as e:
        print(f"ERROR - Music generation failed: {e}")
        raise e


@tool
def generate_background_music_tool(input_params) -> str:
    """
    Generate background music for video content using Meta MusicGen.
    
    Args:
        input_params: Dictionary containing:
            - video_theme: Theme/description of the video (e.g., "Diwali celebration", "fashion showcase")
            - duration: Duration of music in seconds (should match final video duration)
            - custom_prompt: Optional custom music prompt (overrides auto-generated prompt)
    
    Returns:
        JSON string containing music generation results and download URL
    """
    # Extract parameters from input dict
    if isinstance(input_params, str):
        # If it's a JSON string, parse it
        try:
            input_params = json.loads(input_params)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON string: {input_params}")
    
    if not isinstance(input_params, dict):
        raise ValueError(f"Expected dict or JSON string, got {type(input_params)}: {input_params}")
    
    video_theme = input_params.get('video_theme')
    duration = input_params.get('duration')
    custom_prompt = input_params.get('custom_prompt')
    
    if not video_theme:
        raise ValueError("video_theme is required")
    if not duration:
        raise ValueError("duration is required")
    
    print(f"DEBUG - Extracted parameters: video_theme='{video_theme}', duration={duration}, custom_prompt={custom_prompt}")
    start_time = time.time()
    
    # Initialize music_prompt
    music_prompt = ""
    
    try:
        # Generate or use custom prompt
        if custom_prompt:
            music_prompt = custom_prompt
            print(f"DEBUG - Using custom music prompt: {custom_prompt}")
        else:
            music_prompt = generate_music_prompt_with_llm(video_theme)
            print(f"DEBUG - Generated music prompt: {music_prompt}")
        
        # Generate music
        music_url = generate_music_with_replicate(music_prompt, duration)
        
        generation_time = time.time() - start_time
        
        result = GeneratedMusic(
            music_url=music_url,
            prompt=music_prompt,
            duration=duration,
            model_used="meta/musicgen",
            generation_time=generation_time,
            success=True
        )
        
        return result.model_dump_json(indent=2)
        
    except Exception as e:
        generation_time = time.time() - start_time
        
        # Determine prompt for error case
        error_prompt = ""
        if custom_prompt:
            error_prompt = custom_prompt
        elif 'music_prompt' in locals():
            error_prompt = music_prompt
        else:
            error_prompt = generate_music_prompt_with_llm(video_theme)
        
        result = GeneratedMusic(
            music_url="",
            prompt=error_prompt,
            duration=duration,
            model_used="meta/musicgen",
            generation_time=generation_time,
            success=False,
            error_message=str(e)
        )
        
        return result.model_dump_json(indent=2)


def download_music_file(music_url: str, output_path: str) -> bool:
    """Download music file from URL to local path"""
    try:
        print(f"DEBUG - Downloading music from: {music_url}")
        
        import requests
        response = requests.get(music_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"DEBUG - Music downloaded successfully: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR - Failed to download music {music_url}: {e}")
        return False


# Utility function for testing
def test_music_generation():
    """Test function for music generation"""
    print("Testing background music generation...")
    
    # Test Diwali theme
    result = generate_background_music_tool.invoke({
        "video_theme": "Diwali celebration with traditional decorations",
        "duration": 10
    })
    print("Music generation result:")
    print(result)
    
    # Test custom prompt
    custom_result = generate_background_music_tool.invoke({
        "video_theme": "Fashion showcase",
        "duration": 15,
        "custom_prompt": "Upbeat electronic music with fashion show runway vibes, modern and stylish"
    })
    print("\nCustom prompt result:")
    print(custom_result)


if __name__ == "__main__":
    test_music_generation()
