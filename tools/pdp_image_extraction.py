"""
PDP Image Extraction Tool for Pixora AI Agent
Extracts product images from URLs or direct image links
"""
import os
import re
import json
import requests
import tempfile
import base64
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from PIL import Image
import io

def is_image_url(url: str) -> bool:
    """Check if URL points to an image file"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
    parsed = urlparse(url.lower())
    path = parsed.path
    
    # Check for direct image extensions
    if any(path.endswith(ext) for ext in image_extensions):
        return True
    
    # Check for common image hosting services that serve images directly
    image_hosts = ['images.unsplash.com', 'cdn.shopify.com', 'images.pexels.com', 
                   'img.freepik.com', 'i.imgur.com', 'media.istockphoto.com']
    
    if any(host in parsed.netloc for host in image_hosts):
        return True
    
    return False

def extract_urls_from_text(text: str) -> Tuple[List[str], List[str]]:
    """Extract URLs from text and categorize them as image URLs or website URLs"""
    # Regex to find URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    image_urls = []
    website_urls = []
    
    for url in urls:
        if is_image_url(url):
            image_urls.append(url)
        else:
            website_urls.append(url)
    
    return image_urls, website_urls

def download_image(url: str, timeout: int = 30) -> Optional[str]:
    """Download image from URL and save to temporary file"""
    try:
        print(f"DEBUG - Downloading image: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check if response is actually an image
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            print(f"WARNING - URL does not return an image: {url} (content-type: {content_type})")
            return None
        
        # Create temporary file
        suffix = '.jpg'  # Default
        if 'png' in content_type:
            suffix = '.png'
        elif 'gif' in content_type:
            suffix = '.gif'
        elif 'webp' in content_type:
            suffix = '.webp'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            
            print(f"DEBUG - Image downloaded successfully: {tmp_file.name}")
            return tmp_file.name
            
    except Exception as e:
        print(f"ERROR - Failed to download image {url}: {e}")
        return None

def scrape_images_with_gpt(url: str) -> List[str]:
    """Use GPT to scrape images from a webpage"""
    try:
        print(f"DEBUG - Scraping images from: {url}")
        
        # Download webpage content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        html_content = response.text
        
        # Truncate HTML if too long (GPT has token limits)
        if len(html_content) > 50000:
            html_content = html_content[:50000] + "... [truncated]"
        
        # Initialize GPT
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1
        )
        
        # Create prompt for GPT to extract images
        prompt = f"""
You are an expert web scraper. Analyze the following HTML content and extract all product/PDP (Product Detail Page) image URLs.

Focus on:
1. Main product images (hero images, gallery images)
2. Product variant images (different colors, angles, etc.)
3. High-quality product photos
4. Avoid: thumbnails, icons, logos, banners, ads

Return ONLY a JSON array of image URLs. Each URL should be complete and accessible.
Make sure URLs are absolute (include full domain).

HTML Content:
{html_content}

Return format:
["url1", "url2", ...]
"""
        
        # Get response from GPT
        response = llm.invoke(prompt)
        response_text = str(response.content).strip()
        
        print(f"DEBUG - GPT response: {response_text[:200]}...")
        
        # Parse JSON response
        try:
            # Clean up response (remove markdown formatting if present)
            if response_text.startswith('```'):
                response_text = response_text.split('\n', 1)[1]
            if response_text.endswith('```'):
                response_text = response_text.rsplit('\n', 1)[0]
            
            image_urls = json.loads(response_text)
            
            # Validate and clean URLs
            valid_urls = []
            for img_url in image_urls:
                if isinstance(img_url, str):
                    # Make URL absolute if relative
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    elif img_url.startswith('/'):
                        parsed_base = urlparse(url)
                        img_url = f"{parsed_base.scheme}://{parsed_base.netloc}{img_url}"
                    elif not img_url.startswith('http'):
                        img_url = urljoin(url, img_url)
                    
                    valid_urls.append(img_url)
            
            print(f"DEBUG - Extracted {len(valid_urls)} image URLs")
            return valid_urls[:10]  # Limit to 10 images
            
        except json.JSONDecodeError as e:
            print(f"ERROR - Failed to parse GPT response as JSON: {e}")
            print(f"Response was: {response_text}")
            return []
            
    except Exception as e:
        print(f"ERROR - Failed to scrape images from {url}: {e}")
        return []

def convert_image_to_base64(image_path: str) -> Optional[str]:
    """Convert image file to base64 string for display"""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            base64_string = base64.b64encode(img_data).decode('utf-8')
            
            # Determine MIME type
            mime_type = 'image/jpeg'  # Default
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_path.lower().endswith('.gif'):
                mime_type = 'image/gif'
            elif image_path.lower().endswith('.webp'):
                mime_type = 'image/webp'
            
            return f"data:{mime_type};base64,{base64_string}"
            
    except Exception as e:
        print(f"ERROR - Failed to convert image to base64: {e}")
        return None

@tool
def extract_pdp_images_tool(input_text: str, max_images: int = 10) -> str:
    """
    Extract PDP (Product Detail Page) images from user input.
    
    This tool can handle two scenarios:
    1. Direct image URLs provided by user
    2. Website URLs that need to be scraped for product images
    
    Args:
        input_text: User's input text containing URLs or image links
        max_images: Maximum number of images to extract (default: 10)
    
    Returns:
        JSON string containing extracted images and metadata
    """
    
    print(f"DEBUG - extract_pdp_images_tool received:")
    print(f"  input_text length: {len(input_text)}")
    print(f"  max_images: {max_images}")
    
    try:
        # Extract URLs from input text
        image_urls, website_urls = extract_urls_from_text(input_text)
        
        print(f"DEBUG - Found {len(image_urls)} direct image URLs")
        print(f"DEBUG - Found {len(website_urls)} website URLs")
        
        all_image_urls = []
        source_info = []
        
        # Add direct image URLs
        for url in image_urls:
            all_image_urls.append(url)
            source_info.append({"url": url, "source": "direct_link"})
        
        # Scrape images from website URLs
        for url in website_urls:
            scraped_urls = scrape_images_with_gpt(url)
            for scraped_url in scraped_urls:
                all_image_urls.append(scraped_url)
                source_info.append({"url": scraped_url, "source": f"scraped_from_{url}"})
        
        # Limit to max_images
        all_image_urls = all_image_urls[:max_images]
        source_info = source_info[:max_images]
        
        print(f"DEBUG - Total image URLs to process: {len(all_image_urls)}")
        
        if not all_image_urls:
            return json.dumps({
                "success": False,
                "message": "No image URLs found in the input text",
                "extracted_images": [],
                "downloaded_images": [],
                "display_images": [],
                "source_info": []
            })
        
        # Download images
        downloaded_images = []
        display_images = []
        successful_downloads = []
        
        for i, url in enumerate(all_image_urls):
            print(f"DEBUG - Processing image {i+1}/{len(all_image_urls)}: {url}")
            
            # Download image
            local_path = download_image(url)
            if local_path:
                downloaded_images.append(local_path)
                
                # Convert to base64 for display
                base64_data = convert_image_to_base64(local_path)
                if base64_data:
                    display_images.append({
                        "url": url,
                        "local_path": local_path,
                        "base64": base64_data,
                        "source": source_info[i]["source"]
                    })
                    successful_downloads.append(source_info[i])
        
        print(f"DEBUG - Successfully downloaded {len(downloaded_images)} images")
        
        result = {
            "success": True,
            "message": f"Successfully extracted and downloaded {len(downloaded_images)} images",
            "total_found": len(all_image_urls),
            "successfully_downloaded": len(downloaded_images),
            "extracted_images": all_image_urls,
            "downloaded_images": downloaded_images,
            "display_images": display_images,
            "source_info": successful_downloads
        }
        
        return json.dumps(result)
        
    except Exception as e:
        print(f"ERROR - extract_pdp_images_tool failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to extract images: {str(e)}",
            "extracted_images": [],
            "downloaded_images": [],
            "display_images": [],
            "source_info": []
        })

# Test function
def test_extraction():
    """Test the image extraction functionality"""
    
    # Test with direct image URLs
    test_input1 = "Create a video with these images: url1 and url2"
    
    # Test with website URL
    test_input2 = "Make a video from this product page: https://www.amazon.com/dp/B08N5WRWNW"
    
    print("Testing direct image URLs:")
    result1 = extract_pdp_images_tool(test_input1)
    print(json.dumps(json.loads(result1), indent=2))
    
    print("\nTesting website URL:")
    result2 = extract_pdp_images_tool(test_input2)
    print(json.dumps(json.loads(result2), indent=2))

if __name__ == "__main__":
    test_extraction()
