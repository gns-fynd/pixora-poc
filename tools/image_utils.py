"""
Image Upload Utilities - Handle image uploads to FAL CDN
"""
import os
from typing import List, Optional
import fal_client


def upload_image_to_fal(image_path: str) -> str:
    """
    Upload a single image to FAL CDN and return the URL
    
    Args:
        image_path: Local path to the image file
        
    Returns:
        CDN URL of the uploaded image
        
    Raises:
        Exception: If upload fails
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"DEBUG - Uploading image to FAL: {image_path}")
        url = fal_client.upload_file(image_path)
        print(f"DEBUG - Successfully uploaded: {image_path} -> {url}")
        return url
        
    except Exception as e:
        print(f"ERROR - Failed to upload {image_path}: {e}")
        raise


def upload_images_to_fal(image_paths: List[str]) -> List[str]:
    """
    Upload multiple images to FAL CDN and return their URLs
    
    Args:
        image_paths: List of local paths to image files
        
    Returns:
        List of CDN URLs in the same order as input paths
        
    Raises:
        Exception: If any upload fails
    """
    if not image_paths:
        return []
    
    uploaded_urls = []
    
    for i, image_path in enumerate(image_paths):
        try:
            url = upload_image_to_fal(image_path)
            uploaded_urls.append(url)
            print(f"DEBUG - Uploaded image {i+1}/{len(image_paths)}: {url}")
            
        except Exception as e:
            print(f"ERROR - Failed to upload image {i+1}/{len(image_paths)}: {image_path}")
            print(f"ERROR - Details: {e}")
            # Re-raise to stop the process if any upload fails
            raise
    
    print(f"DEBUG - Successfully uploaded all {len(uploaded_urls)} images to FAL")
    return uploaded_urls


def validate_fal_url(url: str) -> bool:
    """
    Validate that a URL is a valid FAL CDN URL
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid FAL URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    # FAL URLs typically start with https://fal-files-prod.s3.amazonaws.com/
    fal_domains = [
        "fal-files-prod.s3.amazonaws.com",
        "fal.media",
        "fal-cdn.com"
    ]
    
    return any(domain in url for domain in fal_domains)


def test_fal_upload():
    """Test function to verify FAL upload functionality"""
    test_image = "logo.png"
    
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found. Skipping test.")
        return
    
    try:
        print("Testing FAL image upload...")
        url = upload_image_to_fal(test_image)
        
        if validate_fal_url(url):
            print(f"✅ FAL upload test successful: {url}")
        else:
            print(f"❌ FAL upload test failed: Invalid URL format: {url}")
            
    except Exception as e:
        print(f"❌ FAL upload test failed: {e}")


if __name__ == "__main__":
    test_fal_upload()
