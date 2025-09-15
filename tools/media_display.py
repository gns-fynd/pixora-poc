"""
Media Display Utilities - For displaying images and videos directly in responses
"""
import base64
import os
import mimetypes
from typing import List, Optional, Dict, Any
import requests
from io import BytesIO


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string for direct display
    
    Args:
        image_path: Path to the image file (local or URL)
    
    Returns:
        Base64 encoded string with data URI prefix, or None if failed
    """
    try:
        # Handle URLs
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path, timeout=30)
            response.raise_for_status()
            image_data = response.content
            
            # Try to determine MIME type from URL or response headers
            content_type = response.headers.get('content-type', '')
            if content_type.startswith('image/'):
                mime_type = content_type
            else:
                # Fallback to guessing from URL
                mime_type, _ = mimetypes.guess_type(image_path)
                if not mime_type or not mime_type.startswith('image/'):
                    mime_type = 'image/jpeg'  # Default fallback
        
        # Handle local files
        else:
            if not os.path.exists(image_path):
                print(f"ERROR - Image file not found: {image_path}")
                return None
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = 'image/jpeg'  # Default fallback
        
        # Encode to base64
        base64_data = base64.b64encode(image_data).decode('utf-8')
        data_uri = f"data:{mime_type};base64,{base64_data}"
        
        print(f"DEBUG - Encoded image to base64: {len(base64_data)} characters")
        return data_uri
        
    except Exception as e:
        print(f"ERROR - Failed to encode image {image_path}: {e}")
        return None


def create_image_display_html(image_url: str, prompt: str = "", max_width: int = 400) -> str:
    """
    Create HTML for displaying an image with optional prompt
    
    Args:
        image_url: Image URL or base64 data URI
        prompt: Optional prompt text to display below image
        max_width: Maximum width for the image display
    
    Returns:
        HTML string for displaying the image
    """
    html = f"""
    <div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 8px;">
        <img src="{image_url}" style="max-width: {max_width}px; width: 100%; height: auto; border-radius: 4px;" />
        {f'<p style="margin-top: 8px; font-size: 14px; color: #666;"><strong>Prompt:</strong> {prompt}</p>' if prompt else ''}
    </div>
    """
    return html


def create_video_display_html(video_url: str, title: str = "", max_width: int = 600) -> str:
    """
    Create HTML for displaying a video
    
    Args:
        video_url: Video URL
        title: Optional title for the video
        max_width: Maximum width for the video player
    
    Returns:
        HTML string for displaying the video
    """
    html = f"""
    <div style="margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px;">
        {f'<h4 style="margin-top: 0; color: #333;">{title}</h4>' if title else ''}
        <video controls style="max-width: {max_width}px; width: 100%; height: auto; border-radius: 4px;">
            <source src="{video_url}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <p style="margin-bottom: 0; font-size: 12px; color: #888;">
            <a href="{video_url}" target="_blank" download>Download Video</a>
        </p>
    </div>
    """
    return html


def format_images_with_prompts(generated_images: List[Dict[str, Any]], display_inline: bool = True) -> str:
    """
    Format generated images with their prompts for display
    
    Args:
        generated_images: List of generated image dictionaries
        display_inline: Whether to display images inline (base64) or as links
    
    Returns:
        Formatted string with images and prompts
    """
    if not generated_images:
        return "No images to display."
    
    formatted_output = []
    
    for i, img_data in enumerate(generated_images):
        scene_id = img_data.get('scene_id', i + 1)
        image_url = img_data.get('image_url', '')
        prompt = img_data.get('prompt', 'No prompt available')
        success = img_data.get('success', True)
        
        if not success:
            formatted_output.append(f"**Scene {scene_id}:** ❌ Generation failed")
            continue
        
        if display_inline and image_url:
            # Try to encode image for inline display
            base64_image = encode_image_to_base64(image_url)
            if base64_image:
                html = create_image_display_html(base64_image, prompt)
                formatted_output.append(f"**Scene {scene_id}:**\n{html}")
            else:
                # Fallback to URL display
                formatted_output.append(f"**Scene {scene_id}:**\n🖼️ [View Image]({image_url})\n**Prompt:** {prompt}")
        else:
            # URL-only display
            formatted_output.append(f"**Scene {scene_id}:**\n🖼️ [View Image]({image_url})\n**Prompt:** {prompt}")
    
    return "\n\n".join(formatted_output)


def format_video_with_details(video_result: Dict[str, Any], display_inline: bool = True) -> str:
    """
    Format video result with details for display
    
    Args:
        video_result: Video generation result dictionary
        display_inline: Whether to display video inline or as link
    
    Returns:
        Formatted string with video and details
    """
    if not video_result.get('success', False):
        return f"❌ Video generation failed: {video_result.get('error_message', 'Unknown error')}"
    
    final_video_url = video_result.get('final_video_url', '')
    fal_video_url = video_result.get('fal_video_url', '')
    total_duration = video_result.get('total_duration', 0)
    total_clips = video_result.get('total_clips', 0)
    processing_time = video_result.get('processing_time', 0)
    
    # Use FAL URL if available, otherwise use final video URL
    display_url = fal_video_url if fal_video_url else final_video_url
    
    if display_inline and display_url:
        html = create_video_display_html(display_url, "Final Merged Video")
        formatted_output = f"""
✅ **Video Generation Complete!**

{html}

**Details:**
- 📹 Total clips merged: {total_clips}
- ⏱️ Total duration: {total_duration:.1f} seconds
- 🕐 Processing time: {processing_time:.1f} seconds
- 🔗 FAL CDN URL: {fal_video_url if fal_video_url else 'Not available'}
"""
    else:
        formatted_output = f"""
✅ **Video Generation Complete!**

🎬 [Download Final Video]({display_url})

**Details:**
- 📹 Total clips merged: {total_clips}
- ⏱️ Total duration: {total_duration:.1f} seconds
- 🕐 Processing time: {processing_time:.1f} seconds
- 🔗 FAL CDN URL: {fal_video_url if fal_video_url else 'Not available'}
"""
    
    return formatted_output


def create_validation_prompt(generated_images: List[Dict[str, Any]]) -> str:
    """
    Create a validation prompt for user approval of generated images
    
    Args:
        generated_images: List of generated image dictionaries
    
    Returns:
        Formatted validation prompt with images
    """
    images_display = format_images_with_prompts(generated_images, display_inline=True)
    
    validation_prompt = f"""
## 🎨 Generated Images for Your Review

{images_display}

---

**Please review the generated images above and let me know:**

✅ **Approve all images** - Type "approve" or "looks good" to proceed with video generation

🔄 **Request changes** - Specify which scenes need changes and what modifications you'd like:
   - Example: "Regenerate scene 2 with better lighting"
   - Example: "Change scene 1 background to outdoor setting"

❌ **Start over** - Type "restart" to create a completely new scene breakdown

What would you like to do next?
"""
    
    return validation_prompt


# Utility functions for testing
def test_media_display():
    """Test media display functionality"""
    print("Testing media display utilities...")
    
    # Test image encoding (with a placeholder)
    sample_images = [
        {
            "scene_id": 1,
            "image_url": "https://via.placeholder.com/400x300/FF0000/FFFFFF?text=Scene+1",
            "prompt": "A beautiful red scene with elegant lighting",
            "success": True
        },
        {
            "scene_id": 2,
            "image_url": "https://via.placeholder.com/400x300/00FF00/FFFFFF?text=Scene+2",
            "prompt": "A vibrant green scene with natural elements",
            "success": True
        }
    ]
    
    # Test image formatting
    formatted_images = format_images_with_prompts(sample_images, display_inline=False)
    print("Formatted images:")
    print(formatted_images)
    
    # Test validation prompt
    validation = create_validation_prompt(sample_images)
    print("\nValidation prompt:")
    print(validation)


if __name__ == "__main__":
    test_media_display()
