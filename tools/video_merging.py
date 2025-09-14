"""
Video Merging Tool - Merges individual video clips into final video with transitions
"""
import json
import os
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Union
from langchain.tools import tool
from pydantic import BaseModel, Field
import requests
import time

# Import image utilities for FAL upload
from .image_utils import upload_images_to_fal


class VideoClip(BaseModel):
    """Individual video clip information"""
    scene_id: int = Field(description="Scene identifier")
    video_url: str = Field(description="URL of the video clip")
    duration: float = Field(description="Duration in seconds")
    transition_type: str = Field(description="Transition to next clip", default="cut")


class MergedVideoResult(BaseModel):
    """Result of video merging process"""
    success: bool = Field(description="Whether merging was successful")
    final_video_url: str = Field(description="URL of the final merged video", default="")
    fal_video_url: str = Field(description="FAL CDN URL of uploaded video", default="")
    total_duration: float = Field(description="Total duration of final video")
    total_clips: int = Field(description="Number of clips merged")
    processing_time: float = Field(description="Time taken to merge")
    error_message: Optional[str] = Field(description="Error message if failed", default=None)


def download_video(url: str, output_path: str) -> bool:
    """Download video from URL to local path"""
    try:
        print(f"DEBUG - Downloading video from: {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"DEBUG - Video downloaded successfully: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR - Failed to download video {url}: {e}")
        return False


def detect_audio_streams(video_path: str) -> bool:
    """Check if video file has audio streams using FFprobe"""
    try:
        # Run ffprobe to get stream information
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            import json
            probe_data = json.loads(result.stdout)

            # Check if any stream is audio
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    print(f"DEBUG - Audio stream detected in {video_path}")
                    return True

            print(f"DEBUG - No audio streams found in {video_path}")
            return False
        else:
            print(f"WARNING - FFprobe failed for {video_path}: {result.stderr}")
            return False

    except Exception as e:
        print(f"ERROR - Failed to probe {video_path}: {e}")
        return False


def get_video_resolution(video_path: str) -> tuple:
    """Get video resolution (width, height) using FFprobe"""
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            import json
            probe_data = json.loads(result.stdout)

            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video":
                    width = stream.get("width", 0)
                    height = stream.get("height", 0)
                    return (width, height)

        return (0, 0)
    except Exception as e:
        print(f"ERROR - Failed to get resolution for {video_path}: {e}")
        return (0, 0)


def create_ffmpeg_filter(clips: List[VideoClip]) -> str:
    """Create FFmpeg filter for merging videos with transitions"""
    if len(clips) == 1:
        return "[0:v][0:a]"
    
    filter_parts = []
    video_inputs = []
    audio_inputs = []
    
    for i, clip in enumerate(clips):
        video_inputs.append(f"[{i}:v]")
        audio_inputs.append(f"[{i}:a]")
        
        # Add transition effects between clips
        if i < len(clips) - 1:
            transition = clip.transition_type
            if transition == "fade":
                # Add fade out to current clip and fade in to next clip
                filter_parts.append(f"[{i}:v]fade=t=out:st={clip.duration-0.5}:d=0.5[v{i}fade];")
                filter_parts.append(f"[{i+1}:v]fade=t=in:st=0:d=0.5[v{i+1}fade];")
            elif transition == "dissolve":
                # Cross-dissolve between clips
                filter_parts.append(f"[{i}:v][{i+1}:v]xfade=transition=dissolve:duration=0.5:offset={clip.duration-0.5}[v{i}dissolve];")
    
    # Concatenate all clips
    video_concat = "".join(video_inputs) + f"concat=n={len(clips)}:v=1:a=0[outv];"
    audio_concat = "".join(audio_inputs) + f"concat=n={len(clips)}:v=0:a=1[outa]"
    
    return "".join(filter_parts) + video_concat + audio_concat


def merge_videos_ffmpeg(clips: List[VideoClip], output_path: str) -> bool:
    """Merge videos using FFmpeg with transitions"""
    temp_dir = None
    try:
        # Create temporary directory for downloaded clips
        temp_dir = tempfile.mkdtemp()
        local_clips = []

        # Download all video clips
        for i, clip in enumerate(clips):
            local_path = os.path.join(temp_dir, f"clip_{i}.mp4")
            if download_video(clip.video_url, local_path):
                local_clips.append(local_path)
                print(f"DEBUG - Downloaded clip {i}: {clip.video_url} -> {local_path}")
            else:
                raise Exception(f"Failed to download clip {i}: {clip.video_url}")

        print(f"DEBUG - Downloaded {len(local_clips)} clips successfully")

        # Check if videos have audio streams and get resolutions
        has_audio = False
        resolutions = []

        for clip_path in local_clips:
            if detect_audio_streams(clip_path):
                has_audio = True

            width, height = get_video_resolution(clip_path)
            resolutions.append((width, height))
            print(f"DEBUG - Video resolution: {width}x{height}")

        print(f"DEBUG - Videos have audio streams: {has_audio}")
        print(f"DEBUG - Video resolutions: {resolutions}")

        # Determine target resolution (use the most common or largest)
        if len(set(resolutions)) == 1:
            # All videos have same resolution
            target_width, target_height = resolutions[0]
            print(f"DEBUG - All videos have same resolution: {target_width}x{target_height}")
        else:
            # Different resolutions - use the most common one
            from collections import Counter
            resolution_counts = Counter(resolutions)
            target_width, target_height = resolution_counts.most_common(1)[0][0]
            print(f"DEBUG - Mixed resolutions, using most common: {target_width}x{target_height}")

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y"]  # -y to overwrite output file

        # Handle concatenation using filter_complex (more reliable than concat protocol)
        if len(clips) == 1:
            # Single clip, just copy
            cmd.extend(["-i", local_clips[0]])
            cmd.extend(["-c", "copy", output_path])
        else:
            # Multiple clips - use filter_complex with concat and scaling
            # Add all input files
            for clip_path in local_clips:
                cmd.extend(["-i", clip_path])

            # Build filter_complex string for concatenation with scaling
            filter_parts = []

            if has_audio:
                # Include both video and audio streams with scaling
                for i in range(len(local_clips)):
                    width, height = resolutions[i]
                    if width != target_width or height != target_height:
                        # Scale video to target resolution
                        filter_parts.append(f"[{i}:v]scale={target_width}:{target_height}[v{i}];")
                        filter_parts.append(f"[v{i}][{i}:a]")
                    else:
                        filter_parts.append(f"[{i}:v][{i}:a]")

                concat_filter = "".join(filter_parts) + f"concat=n={len(local_clips)}:v=1:a=1[v][a]"
                cmd.extend(["-filter_complex", concat_filter])
                cmd.extend(["-map", "[v]", "-map", "[a]"])
                cmd.extend(["-c:v", "libx264", "-c:a", "aac", "-strict", "experimental"])
            else:
                # Video-only concatenation with scaling
                for i in range(len(local_clips)):
                    width, height = resolutions[i]
                    if width != target_width or height != target_height:
                        # Scale video to target resolution
                        filter_parts.append(f"[{i}:v]scale={target_width}:{target_height}[v{i}];")
                        filter_parts.append(f"[v{i}]")
                    else:
                        filter_parts.append(f"[{i}:v]")

                concat_filter = "".join(filter_parts) + f"concat=n={len(local_clips)}:v=1:a=0[v]"
                cmd.extend(["-filter_complex", concat_filter])
                cmd.extend(["-map", "[v]"])
                cmd.extend(["-c:v", "libx264"])

            cmd.extend([output_path])

        print(f"DEBUG - Running FFmpeg command: {' '.join(cmd)}")

        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"DEBUG - Video merging successful: {output_path}")
            # Verify output file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"DEBUG - Output file verified: {os.path.getsize(output_path)} bytes")
                return True
            else:
                print(f"ERROR - Output file missing or empty: {output_path}")
                return False
        else:
            print(f"ERROR - FFmpeg failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"ERROR - Video merging failed: {e}")
        return False
    finally:
        # Cleanup temporary files
        if temp_dir:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"DEBUG - Cleaned up temporary directory: {temp_dir}")
            except:
                pass


def upload_video_to_fal(video_path: str) -> Optional[str]:
    """Upload video to FAL and return CDN URL"""
    try:
        print(f"DEBUG - Uploading video to FAL: {video_path}")
        
        # Use the same FAL upload mechanism as images
        # This is a simplified version - you may need to adapt based on FAL's video upload API
        fal_urls = upload_images_to_fal([video_path])
        
        if fal_urls and len(fal_urls) > 0:
            fal_url = fal_urls[0]
            print(f"DEBUG - Video uploaded to FAL successfully: {fal_url}")
            return fal_url
        else:
            print("ERROR - FAL upload returned no URLs")
            return None
            
    except Exception as e:
        print(f"ERROR - Failed to upload video to FAL: {e}")
        return None


@tool
def merge_videos_tool(
    video_clips: Union[str, List[Dict[str, Any]]],
    output_filename: str = "final_video.mp4"
) -> str:
    """
    Merge individual video clips into a final video with transitions and upload to FAL.
    
    Args:
        video_clips: List of video clip dictionaries or JSON string containing the clips
        output_filename: Name for the output video file
    
    Returns:
        JSON string containing merge results and final video URLs
    """
    import time
    start_time = time.time()
    
    try:
        # Handle both string and list inputs (similar to video generation tool)
        if isinstance(video_clips, str):
            try:
                print(f"DEBUG - merge_videos_tool received string input, parsing JSON")
                parsed_data = json.loads(video_clips)
                
                # Handle nested structure from agent
                if "video_clips" in parsed_data:
                    video_clips = parsed_data["video_clips"]
                else:
                    video_clips = parsed_data
                    
            except json.JSONDecodeError as e:
                return json.dumps({
                    "error": "Invalid JSON format in video_clips",
                    "details": str(e),
                    "received_data": video_clips[:500] + "..." if len(video_clips) > 500 else video_clips
                }, indent=2)
        
        print(f"DEBUG - merge_videos_tool processing {len(video_clips)} clips")
        
        # Validate and parse video clips
        clips = []
        for clip_data in video_clips:
            try:
                # Ensure clip_data is a dictionary and has required fields
                if not isinstance(clip_data, dict):
                    return json.dumps({
                        "error": f"Invalid clip data type: expected dict, got {type(clip_data)}",
                        "clip_data": clip_data
                    }, indent=2)
                
                # Validate required fields
                required_fields = ["scene_id", "video_url", "duration"]
                for field in required_fields:
                    if field not in clip_data:
                        return json.dumps({
                            "error": f"Missing required field '{field}' in clip data",
                            "clip_data": clip_data,
                            "required_fields": required_fields
                        }, indent=2)
                
                clip = VideoClip(**clip_data)
                clips.append(clip)
            except Exception as e:
                return json.dumps({
                    "error": f"Invalid clip data: {e}",
                    "clip_data": clip_data
                }, indent=2)
        
        if not clips:
            return json.dumps({
                "error": "No valid video clips provided"
            }, indent=2)
        
        # Sort clips by scene_id to ensure correct order
        clips.sort(key=lambda x: x.scene_id)
        
        # Create temporary output file
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, output_filename)
        
        print(f"DEBUG - Merging {len(clips)} clips into: {output_path}")
        
        # Merge videos using FFmpeg
        merge_success = merge_videos_ffmpeg(clips, output_path)
        
        if not merge_success:
            return json.dumps({
                "error": "Video merging failed",
                "clips_attempted": len(clips)
            }, indent=2)
        
        # Upload merged video to FAL
        fal_url = upload_video_to_fal(output_path)
        
        # Calculate total duration
        total_duration = sum(clip.duration for clip in clips)
        processing_time = time.time() - start_time
        
        result = MergedVideoResult(
            success=True,
            final_video_url=f"file://{output_path}",  # Local path for immediate access
            fal_video_url=fal_url or "",
            total_duration=total_duration,
            total_clips=len(clips),
            processing_time=processing_time
        )
        
        return result.model_dump_json(indent=2)
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        result = MergedVideoResult(
            success=False,
            total_duration=0,
            total_clips=len(video_clips) if video_clips else 0,
            processing_time=processing_time,
            error_message=str(e)
        )
        
        return result.model_dump_json(indent=2)


@tool
def regenerate_and_merge_videos_tool(
    updated_clips: List[Dict[str, Any]],
    all_clips: List[Dict[str, Any]],
    output_filename: str = "updated_final_video.mp4"
) -> str:
    """
    Update specific video clips and re-merge the entire video.
    
    Args:
        updated_clips: List of updated video clips (with new video_urls)
        all_clips: Complete list of all video clips (original + updated)
        output_filename: Name for the output video file
    
    Returns:
        JSON string containing merge results and final video URLs
    """
    try:
        print(f"DEBUG - Regenerating video with {len(updated_clips)} updated clips")
        
        # Create a mapping of updated clips by scene_id
        updated_map = {clip['scene_id']: clip for clip in updated_clips}
        
        # Merge updated clips into the complete list
        final_clips = []
        for clip in all_clips:
            scene_id = clip['scene_id']
            if scene_id in updated_map:
                # Use updated clip
                final_clips.append(updated_map[scene_id])
                print(f"DEBUG - Using updated clip for scene {scene_id}")
            else:
                # Use original clip
                final_clips.append(clip)
        
        # Call the main merge function
        return merge_videos_tool.invoke({
            "video_clips": final_clips,
            "output_filename": output_filename
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Regenerate and merge failed: {e}",
            "updated_clips_count": len(updated_clips) if updated_clips else 0,
            "total_clips_count": len(all_clips) if all_clips else 0
        }, indent=2)


# Utility function for testing
def test_video_merging():
    """Test function for video merging"""
    sample_clips = [
        {
            "scene_id": 1,
            "video_url": "url1",  # Generic placeholder for testing
            "duration": 10.0,
            "transition_type": "fade"
        },
        {
            "scene_id": 2,
            "video_url": "url2",  # Generic placeholder for testing
            "duration": 10.0,
            "transition_type": "cut"
        }
    ]

    print("Testing video merging...")
    result = merge_videos_tool.invoke({
        "video_clips": sample_clips,
        "output_filename": "test_final_video.mp4"
    })
    print("Merge result:")
    print(result)


if __name__ == "__main__":
    test_video_merging()
