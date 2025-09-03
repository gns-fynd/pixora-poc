# Video Agent POC - Product Requirements Document

## 1. Product Overview

### Vision
Build an AI-powered video generation agent that transforms product images into engaging marketing videos using advanced AI models and ReACT (Reasoning + Acting) architecture.

### Core Value Proposition
- Transform static product images into dynamic video content
- Automate video creation process with intelligent scene planning
- Support both direct image upload and PDP URL scraping
- Generate contextually relevant video content based on user prompts

## 2. System Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Video Agent     │───▶│  External APIs  │
│                 │    │  (ReACT Pattern) │    │                 │
│   - File Upload │    │                  │    │ - OpenAI GPT-4  │
│   - URL Input   │    │ ┌──────────────┐ │    │ - Replicate     │
│   - Progress    │    │ │    Tools     │ │    │ - Firecrawl     │
│   - Results     │    │ │   Registry   │ │    │                 │
└─────────────────┘    │ └──────────────┘ │    └─────────────────┘
                       │                  │
                       │ ┌──────────────┐ │
                       │ │   Memory &   │ │
                       │ │    State     │ │
                       │ └──────────────┘ │
                       └──────────────────┘
```

## 3. User Flow

### Primary User Journey
1. **Input Phase**: User uploads images OR provides PDP URL + video prompt
2. **Processing Phase**: Agent analyzes, plans, edits, and animates content
3. **Output Phase**: User receives final video with download option

### Detailed User Flow
```
User Input → Image Acquisition → Scene Planning → Image Enhancement → Video Generation → Final Assembly → Download
```

## 4. Agent Architecture (ReACT Pattern)

### Core Components
- **LLM**: OpenAI GPT-4 for reasoning and decision making
- **Memory**: LangChain MemorySaver for conversation history
- **Tools**: Custom tools for each pipeline step
- **Prompt Template**: Custom ReACT prompt with video generation context

### Agent Reasoning Flow
```
User Input → Thought → Action → Observation → Thought → Action → ... → Final Answer
```

## 5. Tool Specifications

### Tool 1: `scrape_pdp_images`
**Purpose**: Extract product images from PDP URL using Firecrawl
**Input**: Product detail page URL
**Output**: JSON with extracted image URLs and metadata
**Implementation**: Firecrawl API integration with image filtering

### Tool 2: `analyze_images`
**Purpose**: Analyze images using GPT-4 Vision for content understanding
**Input**: List of image URLs
**Output**: JSON with image analysis and descriptions
**Implementation**: GPT-4 Vision API with structured analysis

### Tool 3: `scene_breakdown`
**Purpose**: Create structured scene breakdown using OpenAI JSON mode
**Input**: User prompt + analyzed images
**Output**: JSON with scene-wise prompts and timing
**Implementation**: GPT-4 with JSON mode for structured output

**Expected JSON Schema**:
```json
{
  "total_scenes": 3,
  "estimated_duration": 15,
  "scenes": [
    {
      "scene_id": 1,
      "source_image_index": 0,
      "image_edit_prompt": "Transform to festive Diwali setting with diyas and rangoli",
      "video_animation_prompt": "Gentle zoom with warm lighting, model holding diya",
      "duration": 5,
      "transition_type": "fade"
    }
  ]
}
```

### Tool 4: `edit_image_nano_banana`
**Purpose**: Edit images using Google's Nano Banana model via Replicate
**Input**: Image URL + edit prompt
**Output**: URL of edited image
**Implementation**: Replicate google/nano-banana model integration

### Tool 5: `animate_image_kling`
**Purpose**: Convert image to video using Kling v2.0 model via Replicate
**Input**: Image URL + video prompt + duration
**Output**: URL of generated video
**Implementation**: Replicate kwaivgi/kling-v2.0 model integration

### Tool 6: `merge_videos`
**Purpose**: Merge multiple video clips into final video using ffmpeg
**Input**: List of video URLs + transition effects
**Output**: URL/path of final merged video
**Implementation**: FFmpeg-python for video processing

## 6. Complete Agent Workflow

### Phase 1: Input Processing
```
Thought: "User provided [images/URL] and wants to create a video about [topic]. 
         I need to first get the images and understand what they contain."

Action: scrape_pdp_images (if URL) OR analyze_images (if direct upload)
Observation: "Found 3 product images: model wearing saree, close-up of fabric, full outfit view"
```

### Phase 2: Scene Planning
```
Thought: "Now I have the images and user's request. I need to create a structured 
         plan for the video with scene-by-scene breakdown."

Action: scene_breakdown
Action Input: {"user_prompt": "Create Diwali ad video", "images": ["url1", "url2", "url3"]}
Observation: "Generated 3-scene plan: festive transformation, product showcase, celebration moment"
```

### Phase 3: Image Enhancement
```
Thought: "I have the scene plan. Now I need to edit each image according to the scene requirements."

Action: edit_image_nano_banana
Action Input: {"image_url": "url1", "edit_prompt": "Transform to Diwali setting with diyas"}
Observation: "Image edited successfully, new URL: edited_image_1.jpg"

[Repeat for each scene]
```

### Phase 4: Video Generation
```
Thought: "Images are ready. Now I need to animate each edited image into video clips."

Action: animate_image_kling
Action Input: {"image_url": "edited_image_1.jpg", "video_prompt": "Gentle zoom with warm lighting", "duration": 5}
Observation: "Video generated successfully: scene_1_video.mp4"

[Repeat for each scene]
```

### Phase 5: Final Assembly
```
Thought: "All scene videos are ready. Time to merge them into the final video."

Action: merge_videos
Action Input: {"video_urls": ["scene_1.mp4", "scene_2.mp4", "scene_3.mp4"], "transitions": ["fade", "fade"]}
Observation: "Final video created: final_diwali_ad.mp4"

Thought: "Video creation complete. I can now present the final result to the user."
Final Answer: "Your Diwali advertisement video has been created successfully! 
              The video features 3 scenes with festive transformations of your product images."
```

## 7. Streamlit UI Design

### Main Interface Components

#### Input Section
- File uploader (multiple images, formats: JPG, PNG, WEBP)
- URL input field (PDP URL validation)
- Text area for video prompt (with examples)
- Submit button with validation

#### Progress Section
- Real-time agent reasoning display
- Progress bar for each phase:
  - Image Acquisition (20%)
  - Scene Planning (40%)
  - Image Enhancement (60%)
  - Video Generation (80%)
  - Final Assembly (100%)
- Current action indicator
- Estimated time remaining

#### Results Section
- Preview of processed images (before/after)
- Final video player with controls
- Download button (MP4 format)
- Processing logs (collapsible)
- Share options

### UI Flow
```
Upload/URL Input → Validation → Processing Display → Results Preview → Download
```

## 8. Technical Implementation

### Project Structure
```
pixora-poc/
├── app.py                          # Streamlit main application
├── requirements.txt                # Dependencies
├── .env.example                   # Environment variables template
├── config/
│   ├── __init__.py
│   ├── settings.py                # Configuration management
│   └── prompts.py                 # Agent prompts and templates
├── agent/
│   ├── __init__.py
│   ├── video_agent.py             # Main ReACT agent implementation
│   ├── memory.py                  # Memory and state management
│   └── tools/
│       ├── __init__.py
│       ├── scraping.py            # Firecrawl PDP scraping
│       ├── scene_breakdown.py     # OpenAI JSON mode planning
│       ├── image_analysis.py      # GPT-4 Vision analysis
│       ├── image_editing.py       # Nano Banana editing
│       ├── video_animation.py     # Kling video generation
│       └── video_merging.py       # FFmpeg video merging
├── utils/
│   ├── __init__.py
│   ├── file_handler.py           # File upload/download utilities
│   ├── api_clients.py            # API client wrappers
│   └── validators.py             # Input validation
├── static/
│   ├── temp/                     # Temporary file storage
│   └── outputs/                  # Final video outputs
└── tests/
    ├── __init__.py
    ├── test_tools.py             # Tool unit tests
    └── test_agent.py             # Agent integration tests
```

### Dependencies
```
streamlit>=1.28.0
langchain>=0.1.0
langchain-openai>=0.1.0
openai>=1.0.0
replicate>=0.15.0
firecrawl-py>=0.0.8
ffmpeg-python>=0.2.0
pillow>=10.0.0
requests>=2.31.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

### Environment Variables
```
OPENAI_API_KEY=your_openai_api_key
REPLICATE_API_TOKEN=your_replicate_token
FIRECRAWL_API_KEY=your_firecrawl_key
TEMP_DIR=./static/temp
OUTPUT_DIR=./static/outputs
```

## 9. API Integration Details

### OpenAI APIs
- **GPT-4**: Agent reasoning and decision making
- **GPT-4 Vision**: Image analysis and captioning
- **GPT-4 JSON Mode**: Structured scene breakdown

### Replicate APIs
- **Nano Banana**: `google/nano-banana:f0a9d34b...` for image editing
- **Kling v2.0**: `kwaivgi/kling-v2.0:03c47b84...` for image-to-video

### Firecrawl API
- **Web Scraping**: Extract images from product detail pages
- **Content Extraction**: Filter and validate product images

## 10. Error Handling Strategy

### Tool-Level Error Handling
- API timeout and retry logic (3 attempts with exponential backoff)
- Graceful degradation for failed operations
- Clear error messages for user feedback
- Fallback mechanisms for critical failures

### Agent-Level Error Handling
- Alternative action paths for failed tools
- State recovery mechanisms
- User notification for critical failures
- Conversation context preservation

### UI-Level Error Handling
- Progress interruption handling
- File cleanup on errors
- User-friendly error messages
- Retry options for failed operations

## 11. Performance Considerations

### Optimization Strategies
- Async API calls where possible
- Image compression for faster processing
- Temporary file cleanup after processing
- Progress caching for long operations
- Parallel processing for independent tasks

### Scalability Considerations
- Stateless tool design for horizontal scaling
- External file storage integration ready (S3, GCS)
- Rate limiting for API calls
- Memory management for large files
- Queue system for batch processing

## 12. Quality Assurance

### Testing Strategy
- Unit tests for individual tools
- Integration tests for agent workflows
- End-to-end tests with sample data
- Performance benchmarking
- Error scenario testing

### Success Metrics
- Video generation success rate (>90%)
- Average processing time (<5 minutes)
- User satisfaction with output quality
- API error rate (<5%)
- System uptime (>99%)

## 13. Security Considerations

### Data Protection
- Temporary file encryption
- Secure API key management
- User data privacy compliance
- File cleanup after processing
- Access logging and monitoring

### API Security
- Rate limiting implementation
- Input validation and sanitization
- Secure credential storage
- API usage monitoring
- Error message sanitization

## 14. Future Enhancements

### Phase 2 Features
- Multiple video format outputs
- Custom transition effects
- Batch processing capabilities
- Video template library
- Advanced editing options

### Phase 3 Features
- Real-time collaboration
- Cloud storage integration
- Advanced analytics
- Custom model training
- Enterprise features

## 15. Implementation Timeline

### Week 1: Foundation
- Project setup and dependencies
- Basic Streamlit UI
- Agent framework implementation

### Week 2: Core Tools
- Image scraping and analysis
- Scene breakdown implementation
- Basic video generation

### Week 3: Integration
- End-to-end workflow
- Error handling and optimization
- Testing and refinement

### Week 4: Polish
- UI/UX improvements
- Performance optimization
- Documentation and deployment

This PRD serves as the comprehensive guide for implementing the Video Agent POC with clear specifications, technical details, and success criteria.
