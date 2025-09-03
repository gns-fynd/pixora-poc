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


class Scene(BaseModel):
    """Individual scene in the video breakdown"""
    scene_id: int = Field(description="Unique identifier for the scene (starting from 1)")
    source_image_index: int = Field(description="Index of the source image (0-based)")
    image_edit_prompt: str = Field(description="Prompt for editing the image for this scene")
    video_animation_prompt: str = Field(description="Prompt for animating the edited image")
    duration: float = Field(description="Duration of the scene in seconds")
    transition_type: str = Field(description="Type of transition to next scene", default="fade")


class SceneBreakdown(BaseModel):
    """Complete scene breakdown for video generation"""
    total_scenes: int = Field(description="Total number of scenes in the video")
    estimated_duration: float = Field(description="Total estimated duration in seconds")
    aspect_ratio: str = Field(description="Recommended aspect ratio")
    scenes: List[Scene] = Field(description="List of individual scenes")


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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
            image_analysis_prompt = f"""Analyze the provided images for creating a video about: {user_prompt}

Describe each image focusing on:
- Visual elements relevant to the video concept
- Composition and framing
- Colors, lighting, mood
- Any text or branding visible
- How it could be used in the video narrative

Format your response as:
Image 0: [detailed description]
Image 1: [detailed description]
etc."""

            # Prepare image data for vision model
            image_contents = []
            for i, image_path in enumerate(image_paths):
                try:
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_to_base64(image_path)}"
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
            scene_prompt = f"""Create a complete scene breakdown for this video request: {user_prompt}

Available images analysis:
{image_analysis.content}

Create a structured breakdown with:
- total_scenes: Number of scenes (2-5)
- estimated_duration: Total duration ({target_duration} seconds)
- aspect_ratio: {aspect_ratio}
- scenes: Array of scene objects

Each scene must have:
- scene_id: Sequential number starting from 1
- source_image_index: Which image to use (0-based, max {num_images-1})
- image_edit_prompt: How to enhance/modify the source image
- video_animation_prompt: Camera movements, effects, transitions  
- duration: Scene length in seconds (3-10 seconds each)
- transition_type: "fade", "cut", or "dissolve"

Make scenes flow together narratively and professionally."""

        else:
            # No images - create conceptual breakdown
            scene_prompt = f"""Create a complete scene breakdown for this video request: {user_prompt}

Create a structured breakdown with:
- total_scenes: Number of scenes (2-5)
- estimated_duration: Total duration ({target_duration} seconds)
- aspect_ratio: {aspect_ratio}
- scenes: Array of scene objects

Each scene must have:
- scene_id: Sequential number starting from 1
- source_image_index: Suggest image index (0, 1, 2, etc.)
- image_edit_prompt: Describe how to create/edit the ideal image
- video_animation_prompt: Camera movements, effects, transitions
- duration: Scene length in seconds (3-10 seconds each)
- transition_type: "fade", "cut", or "dissolve"

Focus on creating a professional, engaging video concept that totals {target_duration} seconds."""

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