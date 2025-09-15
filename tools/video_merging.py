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
from .music_generation import download_music_file


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
    """Download video from URL to local path or copy local file"""
    try:
        print(f"DEBUG - Processing video from: {url}")
        
        # Handle local file URLs
        if url.startswith('file://'):
            local_path = url.replace('file://', '')
            if os.path.exists(local_path):
                import shutil
                shutil.copy2(local_path, output_path)
                print(f"DEBUG - Copied local file: {local_path} -> {output_path}")
                return True
            else:
                print(f"ERROR - Local file not found: {local_path}")
                return False
        
        # Handle regular HTTP/HTTPS URLs
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
    """Merge videos using FFmpeg concat demuxer (more reliable approach)"""
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

        # Determine if we need scaling
        unique_resolutions = set(resolutions)
        needs_scaling = len(unique_resolutions) > 1

        if needs_scaling:
            # Use the most common resolution as target
            from collections import Counter
            resolution_counts = Counter(resolutions)
            target_width, target_height = resolution_counts.most_common(1)[0][0]
            print(f"DEBUG - Mixed resolutions, scaling to: {target_width}x{target_height}")
        else:
            target_width, target_height = resolutions[0]
            print(f"DEBUG - All videos have same resolution: {target_width}x{target_height}")

        # Initialize result variable
        result = None
        
        # Handle single clip case
        if len(clips) == 1:
            if needs_scaling:
                # Scale single clip if needed
                cmd = ["ffmpeg", "-y", "-i", local_clips[0]]
                cmd.extend(["-vf", f"scale={target_width}:{target_height}"])
                cmd.extend(["-c:v", "libx264"])
                if has_audio:
                    cmd.extend(["-c:a", "copy"])
                cmd.append(output_path)
            else:
                # Just copy single clip
                cmd = ["ffmpeg", "-y", "-i", local_clips[0], "-c", "copy", output_path]
            
            print(f"DEBUG - Processing single clip: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        else:
            # Multiple clips - use concat demuxer approach
            if needs_scaling:
                # Pre-scale videos that need it, then concatenate
                scaled_clips = []
                
                for i, clip_path in enumerate(local_clips):
                    width, height = resolutions[i]
                    if width != target_width or height != target_height:
                        # Scale this clip
                        scaled_path = os.path.join(temp_dir, f"scaled_clip_{i}.mp4")
                        scale_cmd = ["ffmpeg", "-y", "-i", clip_path]
                        scale_cmd.extend(["-vf", f"scale={target_width}:{target_height}"])
                        scale_cmd.extend(["-c:v", "libx264"])
                        if has_audio:
                            scale_cmd.extend(["-c:a", "copy"])
                        scale_cmd.append(scaled_path)
                        
                        print(f"DEBUG - Scaling clip {i}: {' '.join(scale_cmd)}")
                        scale_result = subprocess.run(scale_cmd, capture_output=True, text=True, timeout=120)
                        
                        if scale_result.returncode != 0:
                            print(f"ERROR - Failed to scale clip {i}: {scale_result.stderr}")
                            return False
                        
                        scaled_clips.append(scaled_path)
                        print(f"DEBUG - Scaled clip {i} successfully")
                    else:
                        # No scaling needed
                        scaled_clips.append(clip_path)
                
                # Now concatenate the scaled clips
                concat_list_path = os.path.join(temp_dir, "concat_list.txt")
                with open(concat_list_path, 'w') as f:
                    for clip_path in scaled_clips:
                        # Use relative paths and escape single quotes
                        f.write(f"file '{os.path.basename(clip_path)}'\n")
                
                # Change to temp directory for concat to work with relative paths
                original_cwd = os.getcwd()
                os.chdir(temp_dir)
                
                try:
                    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat_list.txt"]
                    cmd.extend(["-c", "copy", output_path])
                    
                    print(f"DEBUG - Concatenating scaled clips: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                finally:
                    os.chdir(original_cwd)
                    
            else:
                # All clips have same resolution - direct concatenation
                concat_list_path = os.path.join(temp_dir, "concat_list.txt")
                with open(concat_list_path, 'w') as f:
                    for clip_path in local_clips:
                        f.write(f"file '{os.path.basename(clip_path)}'\n")
                
                # Change to temp directory for concat
                original_cwd = os.getcwd()
                os.chdir(temp_dir)
                
                try:
                    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat_list.txt"]
                    cmd.extend(["-c", "copy", output_path])
                    
                    print(f"DEBUG - Concatenating clips: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                finally:
                    os.chdir(original_cwd)

        # Check result
        if result and result.returncode == 0:
            print(f"DEBUG - Video merging successful: {output_path}")
            # Verify output file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"DEBUG - Output file verified: {os.path.getsize(output_path)} bytes")
                return True
            else:
                print(f"ERROR - Output file missing or empty: {output_path}")
                return False
        elif result:
            print(f"ERROR - FFmpeg failed: {result.stderr}")
            return False
        else:
            print(f"ERROR - No FFmpeg result available")
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


def add_background_music_to_video(video_path: str, music_url: str, output_path: str, music_volume: float = 0.3) -> bool:
    """Add background music to video using FFmpeg"""
    temp_dir = None
    try:
        # Create temporary directory for music file
        temp_dir = tempfile.mkdtemp()
        music_path = os.path.join(temp_dir, "background_music.mp3")
        
        # Download music file
        if not download_music_file(music_url, music_path):
            print(f"ERROR - Failed to download music from: {music_url}")
            return False
        
        print(f"DEBUG - Adding background music to video")
        print(f"DEBUG - Video: {video_path}")
        print(f"DEBUG - Music: {music_path}")
        print(f"DEBUG - Music volume: {music_volume}")
        
        # Check if video has audio streams
        has_video_audio = detect_audio_streams(video_path)
        
        if has_video_audio:
            # Video has audio - mix with background music
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,      # Video input
                "-i", music_path,      # Music input
                "-filter_complex", f"[1:a]volume={music_volume}[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[audio]",
                "-map", "0:v",         # Use video from first input
                "-map", "[audio]",     # Use mixed audio
                "-c:v", "copy",        # Copy video without re-encoding
                "-c:a", "aac",         # Encode audio as AAC
                "-shortest",           # End when shortest input ends
                output_path
            ]
        else:
            # Video has no audio - just add background music as audio track
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,      # Video input
                "-i", music_path,      # Music input
                "-filter_complex", f"[1:a]volume={music_volume}[audio]",
                "-map", "0:v",         # Use video from first input
                "-map", "[audio]",     # Use background music as audio
                "-c:v", "copy",        # Copy video without re-encoding
                "-c:a", "aac",         # Encode audio as AAC
                "-shortest",           # End when shortest input ends
                output_path
            ]
        
        print(f"DEBUG - FFmpeg command: {' '.join(cmd)}")
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"DEBUG - Successfully added background music to video")
            return True
        else:
            print(f"ERROR - FFmpeg failed to add music: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"ERROR - Failed to add background music: {e}")
        return False
    finally:
        # Cleanup temporary files
        if temp_dir:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"DEBUG - Cleaned up music temp directory: {temp_dir}")
            except:
                pass


@tool
def merge_videos_with_music_tool(input_params) -> str:
    """
    Merge video clips and add background music to the final video.
    
    Args:
        input_params: Dictionary containing:
            - video_clips: List of video clip dictionaries or JSON string containing the clips
            - music_url: URL of the background music file
            - output_filename: Name for the output video file (optional)
            - music_volume: Volume level for background music (0.0 to 1.0, default 0.3)
    
    Returns:
        JSON string containing merge results and final video URLs
    """
    import time
    start_time = time.time()
    
    try:
        # Extract parameters from input dict
        if isinstance(input_params, str):
            # If it's a JSON string, parse it
            try:
                input_params = json.loads(input_params)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string: {input_params}")
        
        if not isinstance(input_params, dict):
            raise ValueError(f"Expected dict or JSON string, got {type(input_params)}: {input_params}")
        
        video_clips = input_params.get('video_clips')
        music_url = input_params.get('music_url')
        output_filename = input_params.get('output_filename', 'final_video_with_music.mp4')
        music_volume = input_params.get('music_volume', 0.3)
        
        if not video_clips:
            raise ValueError("video_clips is required")
        if not music_url:
            raise ValueError("music_url is required")
        
        print(f"DEBUG - Extracted parameters: video_clips={len(video_clips) if isinstance(video_clips, list) else 'string'}, music_url='{music_url}', output_filename='{output_filename}', music_volume={music_volume}")
        # First merge videos without music
        print(f"DEBUG - Step 1: Merging video clips")
        merge_result_json = merge_videos_tool.invoke({
            "video_clips": video_clips,
            "output_filename": "temp_merged_video.mp4"
        })
        
        merge_result = json.loads(merge_result_json)
        
        if not merge_result.get("success"):
            return merge_result_json  # Return the error from video merging
        
        # Get the merged video path
        merged_video_url = merge_result.get("final_video_url", "")
        merged_video_path = merged_video_url.replace("file://", "") if merged_video_url.startswith("file://") else merged_video_url
        
        if not os.path.exists(merged_video_path):
            return json.dumps({
                "error": "Merged video file not found",
                "merged_video_path": merged_video_path
            }, indent=2)
        
        # Create output path for final video with music
        temp_dir = tempfile.mkdtemp()
        final_output_path = os.path.join(temp_dir, output_filename)
        
        print(f"DEBUG - Step 2: Adding background music")
        
        # Add background music to the merged video
        music_success = add_background_music_to_video(
            merged_video_path, 
            music_url, 
            final_output_path, 
            music_volume
        )
        
        if not music_success:
            return json.dumps({
                "error": "Failed to add background music to video",
                "video_path": merged_video_path,
                "music_url": music_url
            }, indent=2)
        
        # Upload final video with music to FAL
        fal_url = upload_video_to_fal(final_output_path)
        
        processing_time = time.time() - start_time
        
        result = MergedVideoResult(
            success=True,
            final_video_url=f"file://{final_output_path}",
            fal_video_url=fal_url or "",
            total_duration=merge_result.get("total_duration", 0),
            total_clips=merge_result.get("total_clips", 0),
            processing_time=processing_time
        )
        
        return result.model_dump_json(indent=2)
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        result = MergedVideoResult(
            success=False,
            total_duration=0,
            total_clips=0,
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
