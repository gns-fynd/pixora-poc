"""
Video Generation Tools - Supports Kling v2.0 and other video models
"""
import json
import os
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.tools import tool
import replicate
from pydantic import BaseModel, Field


class GeneratedVideo(BaseModel):
    """Generated video information"""
    scene_id: int = Field(description="Scene identifier")
    video_url: str = Field(description="URL of the generated video")
    model_used: str = Field(description="Model used for generation")
    prompt: str = Field(description="Prompt used for generation")
    start_image_url: str = Field(description="URL of the start image used")
    generation_time: float = Field(description="Time taken to generate in seconds", default=0.0)
    success: bool = Field(description="Whether generation was successful", default=True)
    error_message: Optional[str] = Field(description="Error message if failed", default=None)


class VideoGenerationResult(BaseModel):
    """Result of video generation process"""
    total_scenes: int = Field(description="Total number of scenes processed")
    successful_generations: int = Field(description="Number of successful generations")
    failed_generations: int = Field(description="Number of failed generations")
    generated_videos: List[GeneratedVideo] = Field(description="List of generated videos")
    total_time: float = Field(description="Total time taken for all generations")
    model_used: str = Field(description="Primary model used")


class VideoModel:
    """Base class for video generation models"""
    
    def __init__(self, model_id: str, default_params: Dict[str, Any]):
        self.model_id = model_id
        self.default_params = default_params
    
    def generate_video(self, prompt: str, start_image_url: str, **kwargs) -> str:
        """Generate video with the model"""
        raise NotImplementedError
    
    def prepare_params(self, prompt: str, start_image_url: str, **kwargs) -> Dict[str, Any]:
        """Prepare parameters for the model"""
        params = self.default_params.copy()
        params.update(kwargs)
        params["prompt"] = prompt
        params["start_image"] = start_image_url
        return params


class KlingV2Model(VideoModel):
    """Kling v2.0 video generation model implementation"""
    
    def __init__(self):
        super().__init__(
            model_id="kwaivgi/kling-v2.0:03c47b845aed8a009e0f83a45be0a2100ca11a7077e667a33224a54e85b2965c",
            default_params={
                "cfg_scale": 0.5,
                "duration": 5,
                "negative_prompt": ""
            }
        )
    
    def generate_video(self, prompt: str, start_image_url: str, **kwargs) -> str:
        """Generate video using Kling v2.0"""
        params = self.prepare_params(prompt, start_image_url, **kwargs)
        
        # Remove any unsupported parameters for Kling
        supported_params = ["prompt", "start_image", "cfg_scale", "duration", "negative_prompt"]
        filtered_params = {k: v for k, v in params.items() if k in supported_params}
        
        # Debug: Print what we're sending to the model
        print(f"DEBUG - Kling v2.0 params: {list(filtered_params.keys())}")
        print(f"DEBUG - Using start_image: {filtered_params.get('start_image', 'None')}")
        print(f"DEBUG - Duration: {filtered_params.get('duration', 5)} seconds")
        
        result = replicate.run(self.model_id, input=filtered_params)
        return str(result)


class KlingProModel(VideoModel):
    """Kling Pro model implementation (placeholder for future)"""
    
    def __init__(self):
        # Note: This is a placeholder for future Kling Pro model
        super().__init__(
            model_id="kwaivgi/kling-pro:placeholder_version_hash",  # Replace when available
            default_params={
                "cfg_scale": 0.7,
                "duration": 10,
                "negative_prompt": ""
            }
        )
    
    def generate_video(self, prompt: str, start_image_url: str, **kwargs) -> str:
        """Generate video using Kling Pro (fallback to v2.0 for now)"""
        # For now, fallback to Kling v2.0 since Pro isn't available yet
        fallback_model = KlingV2Model()
        return fallback_model.generate_video(prompt, start_image_url, **kwargs)


# Model registry for easy extension
VIDEO_MODEL_REGISTRY = {
    "kling-v2": KlingV2Model,
    "kling-pro": KlingProModel,
    # Future models can be added here:
    # "runway": RunwayModel,
    # "pika": PikaModel,
}


def get_video_model(model_name: str) -> VideoModel:
    """Get video model instance by name"""
    model_class = VIDEO_MODEL_REGISTRY.get(model_name.lower())
    if not model_class:
        raise ValueError(f"Unsupported video model: {model_name}. Supported models: {list(VIDEO_MODEL_REGISTRY.keys())}")
    return model_class()


def generate_single_video(scene_data: Dict[str, Any], image_url: str, model_name: str, aspect_ratio: str = "16:9") -> GeneratedVideo:
    """Generate a single video for a scene"""
    import time
    start_time = time.time()
    
    try:
        model = get_video_model(model_name)
        
        # Extract scene information
        scene_id = scene_data.get("scene_id", 0)
        video_animation_prompt = scene_data.get("video_animation_prompt", "")
        duration = scene_data.get("duration", 5)
        
        # Ensure duration is within Kling limits (5 or 10 seconds)
        if duration > 10:
            duration = 10
        elif duration < 5:
            duration = 5
        
        print(f"DEBUG - Scene {scene_id}: Generating video with duration {duration}s")
        print(f"DEBUG - Scene {scene_id}: Using image: {image_url}")
        print(f"DEBUG - Scene {scene_id}: Animation prompt: {video_animation_prompt[:100]}...")
        
        # Generate the video
        video_url = model.generate_video(
            prompt=video_animation_prompt,
            start_image_url=image_url,
            duration=duration,
            cfg_scale=0.5
        )
        
        generation_time = time.time() - start_time
        
        return GeneratedVideo(
            scene_id=scene_id,
            video_url=video_url,
            model_used=model_name,
            prompt=video_animation_prompt,
            start_image_url=image_url,
            generation_time=generation_time,
            success=True
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        return GeneratedVideo(
            scene_id=scene_data.get("scene_id", 0),
            video_url="",
            model_used=model_name,
            prompt=scene_data.get("video_animation_prompt", ""),
            start_image_url=image_url,
            generation_time=generation_time,
            success=False,
            error_message=str(e)
        )


@tool
def generate_scene_videos_tool(
    image_generation_result: Union[str, dict],
    model_name: str = "kling-v2",
    aspect_ratio: str = "16:9"
) -> str:
    """
    Generate videos for all scenes using their generated images from image generation step.
    
    Args:
        image_generation_result: JSON string or dict from generate_scene_images_tool
        model_name: Video generation model to use ("kling-v2" or "kling-pro")
        aspect_ratio: Aspect ratio for videos ("1:1", "9:16", "16:9")
    
    Returns:
        JSON string containing video generation results and video URLs
    """
    import time
    start_time = time.time()
    
    try:
        # Override aspect_ratio with environment variable if available (from agent)
        env_aspect_ratio = os.getenv("PIXORA_ASPECT_RATIO")
        if env_aspect_ratio:
            print(f"DEBUG - Overriding aspect_ratio from {aspect_ratio} to {env_aspect_ratio}")
            aspect_ratio = env_aspect_ratio
        
        # Debug: Print received parameters
        print(f"DEBUG - generate_scene_videos_tool received:")
        print(f"  image_generation_result type: {type(image_generation_result)}")
        print(f"  model_name: {model_name}")
        print(f"  aspect_ratio: {aspect_ratio}")
        
        # Parse the image generation result - handle both string and dict inputs
        if isinstance(image_generation_result, str):
            try:
                image_data = json.loads(image_generation_result)
            except json.JSONDecodeError as e:
                return json.dumps({
                    "error": "Invalid JSON format in image_generation_result",
                    "details": str(e),
                    "received_data": image_generation_result[:500] + "..." if len(image_generation_result) > 500 else image_generation_result
                }, indent=2)
        elif isinstance(image_generation_result, dict):
            image_data = image_generation_result
        else:
            return json.dumps({
                "error": f"Invalid image_generation_result type: {type(image_generation_result)}. Expected string or dict.",
                "received_data": str(image_generation_result)[:500] + "..." if len(str(image_generation_result)) > 500 else str(image_generation_result)
            }, indent=2)

        print(f"DEBUG - Parsed image_data keys: {list(image_data.keys())}")

        # Handle nested structure from agent (common case)
        actual_image_data = None
        if "image_generation_result" in image_data and isinstance(image_data["image_generation_result"], dict):
            print(f"DEBUG - Detected nested structure, extracting image_generation_result")
            actual_image_data = image_data["image_generation_result"]
        else:
            actual_image_data = image_data

        print(f"DEBUG - Actual image data keys: {list(actual_image_data.keys())}")

        # Handle different data structures
        generated_images = []
        scenes = []

        # Check if we have the expected structure with generated_images and scene_breakdown
        if "generated_images" in actual_image_data and "scene_breakdown" in actual_image_data:
            print(f"DEBUG - Using standard structure with generated_images and scene_breakdown")
            generated_images = actual_image_data.get("generated_images", [])
            scene_breakdown = actual_image_data.get("scene_breakdown", {})
            scenes = scene_breakdown.get("scenes", [])

        # Handle the actual structure from agent where scenes contain image URLs directly
        elif "scenes" in actual_image_data:
            print(f"DEBUG - Using agent structure with scenes containing image URLs")
            scenes_data = actual_image_data.get("scenes", [])

            # Convert scenes to the expected format
            for scene in scenes_data:
                scene_id = scene.get("scene_id", 0)
                image_url = scene.get("image_url", "")

                if image_url:
                    # Create a generated image entry
                    generated_images.append({
                        "scene_id": scene_id,
                        "image_url": image_url,
                        "success": True
                    })

                    # Create a scene entry with default video prompt if missing
                    scene_entry = {
                        "scene_id": scene_id,
                        "video_animation_prompt": scene.get("video_animation_prompt", f"Smooth camera movement showcasing the scene with professional lighting and composition"),
                        "duration": scene.get("duration", 5)
                    }
                    scenes.append(scene_entry)

        # Fallback: try to find images in any structure
        if not generated_images:
            print(f"DEBUG - No standard structure found, searching for images...")
            # Look for any array that might contain image data
            for key, value in actual_image_data.items():
                if isinstance(value, list) and value:
                    # Check if items have image_url
                    if isinstance(value[0], dict) and "image_url" in value[0]:
                        print(f"DEBUG - Found images in key: {key}")
                        generated_images = [{"scene_id": i+1, "image_url": item.get("image_url"), "success": True}
                                          for i, item in enumerate(value) if item.get("image_url")]
                        # Create default scenes
                        scenes = [{"scene_id": i+1,
                                 "video_animation_prompt": f"Smooth camera movement for scene {i+1}",
                                 "duration": 5}
                                for i in range(len(generated_images))]
                        break

        if not generated_images:
            return json.dumps({
                "error": "No generated images found in image_generation_result",
                "image_data": actual_image_data,
                "original_structure": image_data,
                "debug_info": "Tried multiple data structure formats but couldn't find images"
            }, indent=2)

        print(f"DEBUG - Found {len(generated_images)} generated images")
        print(f"DEBUG - Found {len(scenes)} scene definitions")
        
        # Create a mapping of scene_id to scene data for video prompts
        scene_map = {scene.get("scene_id"): scene for scene in scenes}
        
        # Generate videos in parallel
        generated_videos = []
        
        # Use ThreadPoolExecutor for parallel generation (no concurrency limit)
        with ThreadPoolExecutor(max_workers=len(generated_images)) as executor:
            # Submit all video generation tasks
            future_to_image = {}
            
            for image_info in generated_images:
                if not image_info.get("success", False):
                    print(f"DEBUG - Skipping failed image for scene {image_info.get('scene_id')}")
                    continue
                
                scene_id = image_info.get("scene_id")
                image_url = image_info.get("image_url", "")
                scene_data = scene_map.get(scene_id, {})

                if not scene_data:
                    print(f"WARNING - No scene data found for scene {scene_id}")
                    continue

                if not image_url:
                    print(f"WARNING - No image URL found for scene {scene_id}")
                    continue

                future = executor.submit(generate_single_video, scene_data, image_url, model_name, aspect_ratio)
                future_to_image[future] = image_info
            
            # Collect results as they complete
            for future in as_completed(future_to_image):
                result = future.result()
                generated_videos.append(result)
        
        # Sort by scene_id to maintain order
        generated_videos.sort(key=lambda x: x.scene_id)
        
        total_time = time.time() - start_time
        successful = sum(1 for video in generated_videos if video.success)
        failed = len(generated_videos) - successful
        
        result = VideoGenerationResult(
            total_scenes=len(generated_videos),
            successful_generations=successful,
            failed_generations=failed,
            generated_videos=generated_videos,
            total_time=total_time,
            model_used=model_name
        )
        
        return result.model_dump_json(indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": "Video generation failed",
            "details": str(e),
            "model_used": model_name
        }, indent=2)


@tool
def regenerate_scene_videos_tool(
    scene_ids: List[int],
    image_generation_result: Union[str, dict],
    model_name: str = "kling-v2",
    aspect_ratio: str = "16:9",
        custom_prompts: Optional[Dict[int, str]] = None
) -> str:
    """
    Regenerate videos for specific scene IDs that user didn't like.
    
    Args:
        scene_ids: List of scene IDs to regenerate (e.g., [1, 3, 5])
        image_generation_result: Original image generation result with scene breakdown
        model_name: Video generation model to use
        aspect_ratio: Aspect ratio for videos
        custom_prompts: Optional custom video prompts for specific scenes (dict mapping scene_id to new_prompt)
    
    Returns:
        JSON string containing regeneration results
    """
    import time
    start_time = time.time()
    
    try:
        # Parse the image generation result
        if isinstance(image_generation_result, str):
            image_data = json.loads(image_generation_result)
        elif isinstance(image_generation_result, dict):
            image_data = image_generation_result
        else:
            return json.dumps({
                "error": f"Invalid image_generation_result type: {type(image_generation_result)}. Expected string or dict.",
                "received_data": str(image_generation_result)
            }, indent=2)

        # Handle nested structure from agent (common case)
        actual_image_data = None
        if "image_generation_result" in image_data and isinstance(image_data["image_generation_result"], dict):
            print(f"DEBUG - Detected nested structure in regenerate tool, extracting image_generation_result")
            actual_image_data = image_data["image_generation_result"]
        else:
            actual_image_data = image_data

        generated_images = actual_image_data.get("generated_images", [])
        scene_breakdown = actual_image_data.get("scene_breakdown", {})
        scenes = scene_breakdown.get("scenes", [])
        
        # Create mappings
        scene_map = {scene.get("scene_id"): scene for scene in scenes}
        image_map = {img.get("scene_id"): img for img in generated_images if img.get("success", False)}
        
        # Find scenes and images to regenerate
        videos_to_regenerate = []
        
        for scene_id in scene_ids:
            scene_data = scene_map.get(scene_id)
            image_info = image_map.get(scene_id)
            
            if not scene_data or not image_info:
                continue
            
            # Use custom prompt if provided
            if custom_prompts and scene_id in custom_prompts:
                scene_data = scene_data.copy()
                scene_data["video_animation_prompt"] = custom_prompts[scene_id]
            
            videos_to_regenerate.append((scene_data, image_info.get("image_url")))
        
        if not videos_to_regenerate:
            return json.dumps({
                "error": f"No valid scenes found for regeneration with IDs: {scene_ids}",
                "available_scene_ids": list(scene_map.keys()),
                "available_image_ids": list(image_map.keys())
            }, indent=2)
        
        # Generate videos in parallel
        generated_videos = []
        
        with ThreadPoolExecutor(max_workers=len(videos_to_regenerate)) as executor:
            future_to_scene = {
                executor.submit(generate_single_video, scene_data, image_url, model_name, aspect_ratio): scene_data
                for scene_data, image_url in videos_to_regenerate
            }
            
            for future in as_completed(future_to_scene):
                result = future.result()
                generated_videos.append(result)
        
        # Sort by scene_id
        generated_videos.sort(key=lambda x: x.scene_id)
        
        total_time = time.time() - start_time
        successful = sum(1 for video in generated_videos if video.success)
        failed = len(generated_videos) - successful
        
        result = VideoGenerationResult(
            total_scenes=len(videos_to_regenerate),
            successful_generations=successful,
            failed_generations=failed,
            generated_videos=generated_videos,
            total_time=total_time,
            model_used=model_name
        )
        
        return result.model_dump_json(indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": "Video regeneration failed",
            "details": str(e),
            "scene_ids_requested": scene_ids,
            "model_used": model_name
        }, indent=2)


# Utility function for testing
def test_video_generation():
    """Test function for video generation with generic placeholder URLs"""
    # Using generic placeholder URLs for testing
    sample_image_result = {
        "total_scenes": 2,
        "successful_generations": 2,
        "failed_generations": 0,
        "generated_images": [
            {
                "scene_id": 1,
                "image_url": "url1",  # Generic placeholder for testing
                "success": True
            },
            {
                "scene_id": 2,
                "image_url": "url2",  # Generic placeholder for testing
                "success": True
            }
        ],
        "scene_breakdown": {
            "scenes": [
                {
                    "scene_id": 1,
                    "video_animation_prompt": "Slow zoom in on the dress details with gentle camera movement",
                    "duration": 5
                },
                {
                    "scene_id": 2,
                    "video_animation_prompt": "Gentle pan around the model with soft lighting",
                    "duration": 10
                }
            ]
        }
    }

    print("Testing video generation with generic placeholder URLs...")
    result = generate_scene_videos_tool.invoke({
        "image_generation_result": json.dumps(sample_image_result),
        "model_name": "kling-v2",
        "aspect_ratio": "9:16"
    })
    print("Video generation result:")
    print(result)


if __name__ == "__main__":
    test_video_generation()
