"""
Image Generation Tools - Supports multiple models with modular architecture
"""
import json
import asyncio
import os
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.tools import tool
import replicate
from pydantic import BaseModel, Field


class GeneratedImage(BaseModel):
    """Generated image information"""
    scene_id: int = Field(description="Scene identifier")
    image_url: str = Field(description="URL of the generated image")
    model_used: str = Field(description="Model used for generation")
    prompt: str = Field(description="Prompt used for generation")
    generation_time: float = Field(description="Time taken to generate in seconds", default=0.0)
    success: bool = Field(description="Whether generation was successful", default=True)
    error_message: Optional[str] = Field(description="Error message if failed", default=None)


class ImageGenerationResult(BaseModel):
    """Result of image generation process"""
    total_scenes: int = Field(description="Total number of scenes processed")
    successful_generations: int = Field(description="Number of successful generations")
    failed_generations: int = Field(description="Number of failed generations")
    generated_images: List[GeneratedImage] = Field(description="List of generated images")
    total_time: float = Field(description="Total time taken for all generations")
    model_used: str = Field(description="Primary model used")


class ImageModel:
    """Base class for image generation models"""
    
    def __init__(self, model_id: str, default_params: Dict[str, Any]):
        self.model_id = model_id
        self.default_params = default_params
    
    def generate_image(self, prompt: str, **kwargs) -> str:
        """Generate image with the model"""
        raise NotImplementedError
    
    def prepare_params(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare parameters for the model"""
        params = self.default_params.copy()
        params.update(kwargs)
        params["prompt"] = prompt
        return params


class NanoBananaModel(ImageModel):
    """Google Nano Banana model implementation"""
    
    def __init__(self):
        super().__init__(
            model_id="google/nano-banana:f0a9d34b12ad1c1cd76269a844b218ff4e64e128ddaba93e15891f47368958a0",
            default_params={
                "output_format": "jpg",
                "quality": 90
            }
        )
    
    def generate_image(self, prompt: str, **kwargs) -> str:
        """Generate image using Nano Banana"""
        params = self.prepare_params(prompt, **kwargs)
        
        # Remove any unsupported parameters
        supported_params = ["prompt", "output_format", "image_input"]
        filtered_params = {k: v for k, v in params.items() if k in supported_params}
        
        # Debug: Print what we're sending to the model
        print(f"DEBUG - Nano Banana params: {list(filtered_params.keys())}")
        if "image_input" in filtered_params:
            print(f"DEBUG - Using image_input for image-to-image generation")
        else:
            print(f"DEBUG - Using text-to-image generation only")
        
        result = replicate.run(self.model_id, input=filtered_params)
        return str(result)


# Model registry for easy extension
MODEL_REGISTRY = {
    "nano-banana": NanoBananaModel,
    # Future models can be added here:
    # "flux": FluxModel,
    # "sdxl": SDXLModel,
    # "kontext": KontextModel,  # Removed - not working
}


def get_model(model_name: str) -> ImageModel:
    """Get model instance by name"""
    model_class = MODEL_REGISTRY.get(model_name.lower())
    if not model_class:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_REGISTRY.keys())}")
    return model_class()


def generate_single_image(scene_data: Dict[str, Any], model_name: str, aspect_ratio: str = "1:1", image_paths: Optional[List[str]] = None) -> GeneratedImage:
    """Generate a single image for a scene"""
    import time
    start_time = time.time()
    
    try:
        model = get_model(model_name)
        
        # Extract scene information
        scene_id = scene_data.get("scene_id", 0)
        image_edit_prompt = scene_data.get("image_edit_prompt", "")
        source_image_index = scene_data.get("source_image_index", 0)
        
        # Generate additional parameters based on aspect ratio
        extra_params = {}
        if aspect_ratio == "9:16":
            extra_params.update({"width": 576, "height": 1024})
        elif aspect_ratio == "1:1":
            extra_params.update({"width": 1024, "height": 1024})
        else:  # 16:9
            extra_params.update({"width": 1024, "height": 576})
        
        # Add source image as reference if available
        if image_paths and len(image_paths) > source_image_index:
            source_image_url = image_paths[source_image_index]
            print(f"DEBUG - Scene {scene_id}: Using source image {source_image_index}: {source_image_url}")
            
            # Check if it's a FAL URL or local path
            if source_image_url.startswith("http"):
                # It's already a URL (FAL CDN), use directly
                extra_params["image_input"] = [source_image_url]
                print(f"DEBUG - Scene {scene_id}: Using FAL URL for image-to-image generation")
            else:
                # It's a local path, convert to data URI as fallback
                try:
                    import base64
                    import mimetypes
                    
                    # Determine MIME type
                    mime_type, _ = mimetypes.guess_type(source_image_url)
                    if not mime_type:
                        mime_type = "image/jpeg"  # Default fallback
                    
                    # Read and encode image as base64
                    with open(source_image_url, "rb") as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        data_uri = f"data:{mime_type};base64,{img_base64}"
                    
                    # Nano Banana expects an array of image inputs
                    extra_params["image_input"] = [data_uri]
                    print(f"DEBUG - Scene {scene_id}: Converted local file to data URI for image-to-image generation")
                    print(f"DEBUG - Scene {scene_id}: Data URI length: {len(data_uri)} characters")
                    
                except Exception as img_error:
                    print(f"WARNING - Scene {scene_id}: Could not load source image {source_image_url}: {img_error}")
                    print(f"DEBUG - Scene {scene_id}: Falling back to text-to-image generation")
        else:
            print(f"DEBUG - Scene {scene_id}: No source image available, using text-to-image generation")
        
        # Generate the image
        image_url = model.generate_image(image_edit_prompt, **extra_params)
        
        generation_time = time.time() - start_time
        
        return GeneratedImage(
            scene_id=scene_id,
            image_url=image_url,
            model_used=model_name,
            prompt=image_edit_prompt,
            generation_time=generation_time,
            success=True
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        return GeneratedImage(
            scene_id=scene_data.get("scene_id", 0),
            image_url="",
            model_used=model_name,
            prompt=scene_data.get("image_edit_prompt", ""),
            generation_time=generation_time,
            success=False,
            error_message=str(e)
        )


@tool
def generate_scene_images_tool(
    scene_breakdown_json: Union[str, dict],
    model_name: str = "nano-banana",
    aspect_ratio: str = "1:1"
) -> str:
    """
    Generate images for all scenes in parallel from scene breakdown JSON.
    
    Args:
        scene_breakdown_json: JSON string or dict from scene_breakdown_tool
        model_name: Image generation model to use ("nano-banana" or "kontext")
        aspect_ratio: Aspect ratio for images ("1:1", "9:16", "16:9")
    
    Returns:
        JSON string containing generation results and image URLs
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
        print(f"DEBUG - generate_scene_images_tool received:")
        print(f"  scene_breakdown_json type: {type(scene_breakdown_json)}")
        print(f"  model_name: {model_name}")
        print(f"  aspect_ratio: {aspect_ratio}")
        
        # Parse the scene breakdown - handle both string and dict inputs
        if isinstance(scene_breakdown_json, str):
            try:
                scene_data = json.loads(scene_breakdown_json)
            except json.JSONDecodeError as e:
                return json.dumps({
                    "error": "Invalid JSON format in scene_breakdown_json",
                    "details": str(e),
                    "received_data": scene_breakdown_json[:500] + "..." if len(scene_breakdown_json) > 500 else scene_breakdown_json
                }, indent=2)
        elif isinstance(scene_breakdown_json, dict):
            scene_data = scene_breakdown_json
        else:
            return json.dumps({
                "error": f"Invalid scene_breakdown_json type: {type(scene_breakdown_json)}. Expected string or dict.",
                "received_data": str(scene_breakdown_json)[:500] + "..." if len(str(scene_breakdown_json)) > 500 else str(scene_breakdown_json)
            }, indent=2)
        
        print(f"DEBUG - Parsed scene_data keys: {list(scene_data.keys())}")
        
        # Handle different possible structures
        scenes = []
        image_paths = None
        
        # Check for direct structure first
        if "scenes" in scene_data:
            scenes = scene_data.get("scenes", [])
            image_paths = scene_data.get("image_paths", None)
            print(f"DEBUG - Found scenes in direct structure: {len(scenes)} scenes")
            print(f"DEBUG - Found image_paths in direct structure: {image_paths}")
        
        # Check for nested structure (from agent tool calls)
        elif "scene_breakdown_json" in scene_data and isinstance(scene_data["scene_breakdown_json"], dict):
            actual_breakdown = scene_data["scene_breakdown_json"]
            scenes = actual_breakdown.get("scenes", [])
            image_paths = actual_breakdown.get("image_paths", None)
            print(f"DEBUG - Found scenes in nested structure: {len(scenes)} scenes")
            print(f"DEBUG - Found image_paths in nested structure: {image_paths}")
        
        # Handle case where the entire input is the breakdown
        elif isinstance(scene_data, dict) and all(key in scene_data for key in ["total_scenes", "scenes"]):
            scenes = scene_data.get("scenes", [])
            image_paths = scene_data.get("image_paths", None)
            print(f"DEBUG - Input is direct breakdown: {len(scenes)} scenes")
            print(f"DEBUG - Found image_paths in breakdown: {image_paths}")
        
        print(f"DEBUG - Found {len(scenes)} scenes")
        
        if not scenes:
            print(f"DEBUG - scene_data content: {scene_data}")
            return json.dumps({
                "error": "No scenes found in breakdown",
                "scene_breakdown": scene_data,
                "debug_info": {
                    "top_level_keys": list(scene_data.keys()),
                    "has_nested_breakdown": "scene_breakdown_json" in scene_data,
                    "nested_keys": list(scene_data.get("scene_breakdown_json", {}).keys()) if isinstance(scene_data.get("scene_breakdown_json"), dict) else "Not a dict"
                }
            }, indent=2)
        
        # Generate images in parallel
        generated_images = []
        
        # Use ThreadPoolExecutor for parallel generation
        with ThreadPoolExecutor(max_workers=min(len(scenes), 5)) as executor:
            # Submit all generation tasks with image paths
            future_to_scene = {
                executor.submit(generate_single_image, scene, model_name, aspect_ratio, image_paths): scene
                for scene in scenes
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_scene):
                result = future.result()
                generated_images.append(result)
        
        # Sort by scene_id to maintain order
        generated_images.sort(key=lambda x: x.scene_id)
        
        total_time = time.time() - start_time
        successful = sum(1 for img in generated_images if img.success)
        failed = len(generated_images) - successful
        
        result = ImageGenerationResult(
            total_scenes=len(scenes),
            successful_generations=successful,
            failed_generations=failed,
            generated_images=generated_images,
            total_time=total_time,
            model_used=model_name
        )
        
        return result.model_dump_json(indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": "Image generation failed",
            "details": str(e),
            "model_used": model_name
        }, indent=2)


@tool
def regenerate_scene_images_tool(
    scene_ids: List[int],
    scene_breakdown_json: Union[str, dict],
    model_name: str = "nano-banana",
    aspect_ratio: str = "1:1",
    custom_prompts: Optional[Dict[int, str]] = None
) -> str:
    """
    Regenerate images for specific scene IDs that user didn't like.
    
    Args:
        scene_ids: List of scene IDs to regenerate (e.g., [1, 3, 5])
        scene_breakdown_json: Original scene breakdown JSON
        model_name: Image generation model to use
        aspect_ratio: Aspect ratio for images
        custom_prompts: Optional custom prompts for specific scenes (dict mapping scene_id to new_prompt)
    
    Returns:
        JSON string containing regeneration results
    """
    import time
    start_time = time.time()
    
    try:
        # Parse the scene breakdown - handle both string and dict inputs
        if isinstance(scene_breakdown_json, str):
            scene_data = json.loads(scene_breakdown_json)
        elif isinstance(scene_breakdown_json, dict):
            scene_data = scene_breakdown_json
        else:
            return json.dumps({
                "error": f"Invalid scene_breakdown_json type: {type(scene_breakdown_json)}. Expected string or dict.",
                "received_data": str(scene_breakdown_json)
            }, indent=2)
        
        scenes = scene_data.get("scenes", [])
        image_paths = scene_data.get("image_paths", None)
        
        # Find scenes to regenerate
        scenes_to_regenerate = []
        for scene in scenes:
            if scene.get("scene_id") in scene_ids:
                # Use custom prompt if provided
                if custom_prompts and scene.get("scene_id") in custom_prompts:
                    scene = scene.copy()
                    scene["image_edit_prompt"] = custom_prompts[scene.get("scene_id")]
                scenes_to_regenerate.append(scene)
        
        if not scenes_to_regenerate:
            return json.dumps({
                "error": f"No scenes found with IDs: {scene_ids}",
                "available_scene_ids": [s.get("scene_id") for s in scenes]
            }, indent=2)
        
        # Generate images in parallel
        generated_images = []
        
        with ThreadPoolExecutor(max_workers=min(len(scenes_to_regenerate), 5)) as executor:
            future_to_scene = {
                executor.submit(generate_single_image, scene, model_name, aspect_ratio, image_paths): scene
                for scene in scenes_to_regenerate
            }
            
            for future in as_completed(future_to_scene):
                result = future.result()
                generated_images.append(result)
        
        # Sort by scene_id
        generated_images.sort(key=lambda x: x.scene_id)
        
        total_time = time.time() - start_time
        successful = sum(1 for img in generated_images if img.success)
        failed = len(generated_images) - successful
        
        result = ImageGenerationResult(
            total_scenes=len(scenes_to_regenerate),
            successful_generations=successful,
            failed_generations=failed,
            generated_images=generated_images,
            total_time=total_time,
            model_used=model_name
        )
        
        return result.model_dump_json(indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": "Scene regeneration failed",
            "details": str(e),
            "scene_ids_requested": scene_ids,
            "model_used": model_name
        }, indent=2)


# Utility function for testing
def test_image_generation():
    """Test function for image generation"""
    sample_scene_breakdown = {
        "total_scenes": 2,
        "estimated_duration": 10,
        "aspect_ratio": "1:1",
        "scenes": [
            {
                "scene_id": 1,
                "source_image_index": 0,
                "image_edit_prompt": "A professional product shot of a dress on a mannequin in a modern studio",
                "video_animation_prompt": "Slow zoom in on the dress details",
                "duration": 5,
                "transition_type": "fade"
            },
            {
                "scene_id": 2,
                "source_image_index": 1,
                "image_edit_prompt": "A model wearing the dress in an elegant pose with soft lighting",
                "video_animation_prompt": "Gentle pan around the model",
                "duration": 5,
                "transition_type": "cut"
            }
        ]
    }
    
    # Test bulk generation
    print("Testing bulk generation...")
    result = generate_scene_images_tool.invoke({
        "scene_breakdown_json": json.dumps(sample_scene_breakdown),
        "model_name": "nano-banana",
        "aspect_ratio": "1:1"
    })
    print("Bulk generation result:")
    print(result)
    
    # Test regeneration
    print("\nTesting regeneration...")
    regen_result = regenerate_scene_images_tool.invoke({
        "scene_ids": [1],
        "scene_breakdown_json": json.dumps(sample_scene_breakdown),
        "model_name": "nano-banana",
        "aspect_ratio": "1:1",
        "custom_prompts": {1: "A stunning product shot with dramatic lighting"}
    })
    print("Regeneration result:")
    print(regen_result)


if __name__ == "__main__":
    test_image_generation()
