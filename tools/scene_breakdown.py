"""
Scene Breakdown Tool - Uses LangChain's structured output to create scene breakdowns
"""
import json
import base64
from typing import List, Dict, Any, Optional
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import os
from pydantic import BaseModel, Field

# Import prompts
from config.prompts import (
    SCENE_BREAKDOWN_WITH_IMAGES,
    SCENE_BREAKDOWN_NO_IMAGES,
    IMAGE_ANALYSIS_PROMPT
)

# Import image utilities
from .image_utils import upload_images_to_fal


class Scene(BaseModel):
    """Individual scene in the video breakdown"""
    scene_id: int = Field(description="Unique identifier for the scene (starting from 1)")
    source_image_index: int = Field(description="Index of the source image (0-based)")
    image_edit_prompt: str = Field(description="Prompt for editing the image for this scene - must mention character and garment consistency")
    video_animation_prompt: str = Field(description="Prompt for animating the edited image (5-10 seconds max)")
    duration: float = Field(description="Duration of the scene in seconds (5 or 10 seconds only)", ge=5, le=10)
    transition_type: str = Field(description="Type of transition to next scene", default="fade")


class SceneBreakdown(BaseModel):
    """Complete scene breakdown for video generation"""
    total_scenes: int = Field(description="Total number of scenes in the video")
    estimated_duration: float = Field(description="Total estimated duration in seconds")
    aspect_ratio: str = Field(description="Recommended aspect ratio")
    image_paths: Optional[List[str]] = Field(description="List of source image paths", default=None)
    scenes: List[Scene] = Field(description="List of individual scenes")


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Convert image file to base64 string with proper MIME type detection
    
    Returns:
        tuple: (base64_string, mime_type)
    """
    import mimetypes
    from PIL import Image
    
    try:
        # Validate file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Get file size for validation
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            raise ValueError(f"Image file is empty: {image_path}")
        
        print(f"DEBUG - Processing image: {image_path} (size: {file_size} bytes)")
        
        # Try to detect MIME type from file extension
        mime_type, _ = mimetypes.guess_type(image_path)
        
        # Validate the image using PIL to ensure it's not corrupted
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify the image is valid
                image_format = img.format.lower() if img.format else None
                print(f"DEBUG - Image format detected: {image_format}")
        except Exception as pil_error:
            print(f"WARNING - PIL validation failed for {image_path}: {pil_error}")
            # Continue anyway, might still work
        
        # Map common formats to proper MIME types
        if not mime_type:
            extension = os.path.splitext(image_path)[1].lower()
            mime_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg', 
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_map.get(extension, 'image/jpeg')  # Default to jpeg
        
        print(f"DEBUG - Using MIME type: {mime_type}")
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            if len(image_data) == 0:
                raise ValueError(f"No data read from image file: {image_path}")
            
            base64_string = base64.b64encode(image_data).decode('utf-8')
            print(f"DEBUG - Base64 encoded successfully, length: {len(base64_string)}")
            
            return base64_string, mime_type
            
    except Exception as e:
        print(f"ERROR - Failed to encode image {image_path}: {e}")
        raise


@tool
def scene_breakdown_tool(
    user_prompt: str,
    image_paths: Optional[List[str]] = None,
    aspect_ratio: str = "16:9",
    duration_preference: str = "30sec"
) -> str:
    """
    Analyze user prompt and images to create a structured scene breakdown for video generation.
    
    Args:
        user_prompt: The user's description of what video they want
        image_paths: List of paths to uploaded images (optional)
        aspect_ratio: Preferred aspect ratio (1:1, 9:16, etc.)
        duration_preference: Preferred total duration (10sec, 30sec, 1min)
    
    Returns:
        JSON string containing structured scene breakdown
    """
    
    # Debug: Print received parameters before override
    print(f"DEBUG - scene_breakdown_tool received:")
    print(f"  aspect_ratio: {aspect_ratio} (type: {type(aspect_ratio)})")
    print(f"  duration_preference: {duration_preference} (type: {type(duration_preference)})")
    print(f"  user_prompt length: {len(user_prompt)}")
    print(f"  image_paths: {image_paths}")
    print(f"  num_images: {len(image_paths) if image_paths else 0}")
    print(f"  All function args received: user_prompt={user_prompt is not None}, image_paths={image_paths is not None}, aspect_ratio={aspect_ratio}, duration_preference={duration_preference}")
    
    # Override parameters with environment variables if available (from agent)
    env_image_paths = os.getenv("PIXORA_IMAGE_PATHS")
    env_aspect_ratio = os.getenv("PIXORA_ASPECT_RATIO")
    env_duration = os.getenv("PIXORA_DURATION")
    
    print(f"DEBUG - Before override: aspect_ratio={aspect_ratio}, duration={duration_preference}")
    
    if env_image_paths and env_image_paths != "None":
        try:
            # Parse the string representation of the list
            import ast
            image_paths = ast.literal_eval(env_image_paths)
            print(f"DEBUG - Using image_paths from environment: {image_paths}")
        except Exception as e:
            print(f"WARNING - Could not parse image_paths from environment: {e}")
    
    # Only override if environment variables are provided AND different from user input
    if env_aspect_ratio and env_aspect_ratio != aspect_ratio:
        print(f"DEBUG - Environment override: aspect_ratio from {aspect_ratio} to {env_aspect_ratio}")
        aspect_ratio = env_aspect_ratio
    
    if env_duration and env_duration != duration_preference:
        print(f"DEBUG - Environment override: duration_preference from {duration_preference} to {env_duration}")
        duration_preference = env_duration
    
    print(f"DEBUG - After override: aspect_ratio={aspect_ratio}, duration={duration_preference}")
    
    # Debug: Print final parameters
    print(f"DEBUG - scene_breakdown_tool final parameters:")
    print(f"  aspect_ratio: {aspect_ratio} (type: {type(aspect_ratio)})")
    print(f"  duration_preference: {duration_preference} (type: {type(duration_preference)})")
    print(f"  user_prompt length: {len(user_prompt)}")
    print(f"  image_paths: {image_paths}")
    print(f"  num_images: {len(image_paths) if image_paths else 0}")
    
    # Convert duration preference to seconds
    duration_map = {
        "10sec": 10,
        "30sec": 30,
        "1min": 60
    }
    target_duration = duration_map.get(duration_preference, 30)
    
    # Check if images are provided
    num_images = len(image_paths) if image_paths else 0
    
    try:
        # Initialize LangChain ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create structured LLM that returns SceneBreakdown objects
        # Note: Using function calling instead of json_mode for better reliability
        structured_llm = llm.with_structured_output(SceneBreakdown)
        
        if num_images > 0:
            # Step 1: Analyze images using regular LLM (structured output doesn't support vision)
            image_analysis_prompt = IMAGE_ANALYSIS_PROMPT.format(user_prompt=user_prompt)

            # Prepare image data for vision model
            image_contents = []
            for i, image_path in enumerate(image_paths):
                try:
                    base64_data, mime_type = encode_image_to_base64(image_path)
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_data}"
                        }
                    })
                except Exception as e:
                    print(f"Warning: Could not load image {image_path}: {e}")
                    continue

            # Get image analysis
            messages = [
                {"role": "system", "content": image_analysis_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"User wants to create: {user_prompt}"},
                    *image_contents
                ]}
            ]
            
            image_analysis = llm.invoke(messages)
            
            # Step 2: Create scene breakdown with structured output
            scene_prompt = SCENE_BREAKDOWN_WITH_IMAGES.format(
                user_prompt=user_prompt,
                image_analysis=image_analysis.content,
                target_duration=target_duration,
                aspect_ratio=aspect_ratio,
                max_image_index=num_images-1
            )

        else:
            # No images - create conceptual breakdown
            scene_prompt = SCENE_BREAKDOWN_NO_IMAGES.format(
                user_prompt=user_prompt,
                target_duration=target_duration,
                aspect_ratio=aspect_ratio
            )

        # Get structured scene breakdown
        scene_breakdown = structured_llm.invoke(scene_prompt)
        
        # Validate and ensure all required fields are present
        if not isinstance(scene_breakdown, SceneBreakdown):
            # If we didn't get a proper SceneBreakdown object, try to create one
            if isinstance(scene_breakdown, dict):
                try:
                    scene_breakdown = SceneBreakdown(**scene_breakdown)
                except Exception as e:
                    return json.dumps({
                        "error": "Schema validation failed",
                        "details": str(e),
                        "received_data": scene_breakdown
                    }, indent=2)
        
        # Validate that LLM respected user configuration
        print(f"DEBUG - LLM output: aspect_ratio={scene_breakdown.aspect_ratio}, duration={scene_breakdown.estimated_duration}")
        print(f"DEBUG - User requested: aspect_ratio={aspect_ratio}, duration={target_duration}")
        
        # Only override if LLM completely ignored user preferences
        if scene_breakdown.aspect_ratio != aspect_ratio:
            print(f"DEBUG - Correcting aspect_ratio from {scene_breakdown.aspect_ratio} to {aspect_ratio}")
            scene_breakdown.aspect_ratio = aspect_ratio
        
        if abs(scene_breakdown.estimated_duration - target_duration) > 5:  # Allow 5 second tolerance
            print(f"DEBUG - Correcting duration from {scene_breakdown.estimated_duration} to {target_duration}")
            scene_breakdown.estimated_duration = float(target_duration)
        
        # Validate scene durations add up reasonably to target duration
        total_scene_duration = sum(scene.duration for scene in scene_breakdown.scenes)
        if abs(total_scene_duration - target_duration) > 10:  # Allow 10 second tolerance
            # Adjust scene durations proportionally to match target
            scale_factor = target_duration / total_scene_duration
            for scene in scene_breakdown.scenes:
                new_duration = scene.duration * scale_factor
                # Round to nearest 5 or 10 seconds
                scene.duration = 10.0 if new_duration > 7.5 else 5.0
            
            # Recalculate total duration
            scene_breakdown.estimated_duration = float(sum(scene.duration for scene in scene_breakdown.scenes))
        
        # Upload images to FAL and store URLs instead of local paths
        if image_paths and num_images > 0:
            try:
                print(f"DEBUG - Uploading {num_images} images to FAL...")
                fal_urls = upload_images_to_fal(image_paths)
                scene_breakdown.image_paths = fal_urls
                print(f"DEBUG - Successfully uploaded all images to FAL")
            except Exception as upload_error:
                print(f"WARNING - Failed to upload images to FAL: {upload_error}")
                print(f"DEBUG - Falling back to local paths (image-to-image may not work)")
                scene_breakdown.image_paths = image_paths
        else:
            scene_breakdown.image_paths = None
        
        # Return as JSON string
        return scene_breakdown.model_dump_json(indent=2)
        
    except Exception as api_error:
        return json.dumps({
            "error": "Scene breakdown generation failed",
            "details": str(api_error),
            "user_prompt": user_prompt,
            "config": {
                "aspect_ratio": aspect_ratio,
                "duration": duration_preference,
                "num_images": num_images
            }
        }, indent=2)


# Example usage function for testing
def test_scene_breakdown():
    """Test function to demonstrate scene breakdown tool"""
    sample_prompt = "Create a festive Diwali advertisement video showcasing traditional clothing with warm lighting and celebration themes"
    
    result = scene_breakdown_tool(
        user_prompt=sample_prompt,
        image_paths=None,  # Test without images
        aspect_ratio="9:16",
        duration_preference="30sec"
    )
    
    print("Scene Breakdown Result:")
    print(result)


if __name__ == "__main__":
    test_scene_breakdown()
