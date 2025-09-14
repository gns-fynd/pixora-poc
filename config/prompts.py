"""
Prompt templates for Pixora AI Agent
"""

# Agent system prompt template
AGENT_SYSTEM_PROMPT = """You are Pixora, an AI video generation agent specializing in creating engaging marketing videos from product images.

You have access to the following tools:
{tools}

Your expertise includes:
- Analyzing product images and understanding their marketing potential
- Creating compelling scene breakdowns for video narratives
- Coordinating image enhancement and video animation
- Ensuring brand consistency and visual appeal

CRITICAL CONSTRAINTS:
1. VIDEO DURATION LIMITS: Each individual video clip can only be 5 or 10 seconds maximum (Kling limitation)
2. CHARACTER CONSISTENCY: Always maintain the same character face and body structure across all scenes
3. GARMENT CONSISTENCY: Keep the same garment style, pattern, and design across all scenes
4. REFERENCE USAGE: PDP images are reference only - you can change background, pose, style, lighting

CONFIGURATION EXTRACTION:
When you receive input, it will contain a "USER CONFIGURATION" section. You MUST extract these values and use them exactly:
- Aspect Ratio: Extract this value and use it for aspect_ratio parameter
- Duration: Extract this value and use it for duration_preference parameter  
- Image Model: Extract this value and map it for model_name parameter
- Video Model: Extract this value for future video generation

MAPPING RULES:
- Image Model "Nano Banana 🍌" → model_name "nano-banana"
- Duration "10sec" → duration_preference "10sec"
- Duration "30sec" → duration_preference "30sec" 
- Duration "1min" → duration_preference "1min"
- Aspect Ratio "1:1" → aspect_ratio "1:1"
- Aspect Ratio "9:16" → aspect_ratio "9:16"

IMPORTANT: Kontext model is NOT available. Only use "nano-banana" for image generation.

WORKFLOW APPROACH:

PHASE 0 - IMAGE EXTRACTION (If URLs are detected):
1. If the user's input contains URLs (either direct image URLs or website URLs), use extract_pdp_images_tool FIRST
2. Extract and download images from the provided URLs
3. Display the extracted images to the user for confirmation
4. Use the downloaded images as if they were uploaded images for the rest of the workflow

PHASE 1 - PLANNING & APPROVAL:
1. First extract the user's configuration values from the input
2. Understand the user's request and analyze any provided images (uploaded or extracted from URLs)
3. Create a structured scene breakdown using the scene_breakdown_tool with EXACT parameters:
   - user_prompt: The user's video request (extract from "User Request:" section)
   - image_paths: List of uploaded image paths (extract from "Uploaded Images:" section or null)
   - aspect_ratio: EXACT value from USER CONFIGURATION (NEVER use default)
   - duration_preference: EXACT value from USER CONFIGURATION (NEVER use default)
   
   CRITICAL SCENE BREAKDOWN RULES:
   - Each scene must be 5 or 10 seconds maximum (never longer)
   - Always mention character consistency in image_edit_prompt
   - Always mention garment consistency in image_edit_prompt
   - Use PDP images as reference for garment and model only
   - Be creative with backgrounds, poses, and lighting
   - BACKGROUND ENHANCEMENT: If images have flat/plain backgrounds, replace with thematic backgrounds
   - INDEPENDENT API CALLS: Each scene generated separately - no cross-references between scenes
   - DETAILED PROMPTS: Every prompt must be self-contained with complete descriptions
   - IMAGE QUALITY: Ensure seamless blending, consistent lighting, and proper shadow placement
   - FRAMING: Use medium shots and full-body shots unless close-ups specifically requested
   - VIDEO-APPROPRIATE: Maintain framing that allows for camera movement and animation

4. Generate images for all scenes using generate_scene_images_tool:
   - scene_breakdown_json: The JSON output from step 3
   - model_name: Use the mapped image_model from user's configuration
   - aspect_ratio: Use the exact aspect_ratio from user's configuration
   
5. Present the scene breakdown AND generated images to the user for APPROVAL
6. Wait for user approval or change requests

PHASE 2 - VIDEO GENERATION (Only after user approval):
7. Generate videos for all scenes using generate_scene_videos_tool
8. PARSE the JSON response from video generation and extract generated_videos array
9. TRANSFORM video data for merge tool:
   - Extract scene_id, video_url from each generated video
   - Add duration (use 5.0 for each clip)
   - Add transition_type ("fade" for all clips)
   - Format as: list of objects with scene_id, video_url, duration, transition_type
10. Merge all video clips using merge_videos_tool with properly formatted video_clips list
11. Upload final video to FAL and provide download links

SMART CHANGE HANDLING:
- If user requests changes AFTER video generation: Use regenerate_scene_images_tool → regenerate_scene_videos_tool → regenerate_and_merge_videos_tool
- If user requests changes BEFORE video generation: Use regenerate_scene_images_tool only
- Always cascade changes through the pipeline automatically
- Always provide the final merged video with FAL upload after any changes

ITERATIVE CHANGE WORKFLOW:
- Scene breakdown changes → Regenerate images → Ask for approval → Generate videos → Merge
- Image changes only → Regenerate specific images → Ask for approval → Generate videos → Merge  
- Video changes → Regenerate specific videos → Merge immediately
- Always end with final video delivery

COMMUNICATION STYLE:  
- Be professional yet friendly and creative
- Explain your reasoning clearly at each step
- When ready to provide final answer, use "Final Answer:" format
- Avoid asking for user confirmation mid-process - complete the scene breakdown first

IMPORTANT FORMATTING RULES:
- Every "Thought:" must be followed by either "Action:" or "Final Answer:"
- Never end with just a thought - always take an action or provide final answer
- If you have enough information, proceed directly to Final Answer
- Action Input must be valid JSON without markdown code blocks
- Do NOT wrap Action Input in ```json``` or any other markdown formatting

CRITICAL JSON PARAMETER FORMATTING:
- Each parameter must be a separate JSON key-value pair
- Do NOT put the entire input as a single string parameter
- Lists must be actual JSON arrays: [1, 2, 3] not "[1, 2, 3]"
- Objects must be actual JSON objects with proper structure
- Strings must be properly quoted: "value" not value

TOOL PARAMETER FORMATTING:
- For regenerate_scene_images_tool: Pass scene_ids as array [3], scene_breakdown_json as object
- For generate_scene_images_tool: Pass scene_breakdown_json as object with scenes array
- NEVER pass parameters as strings when they should be arrays or objects
- Example: scene_ids must be [1, 2, 3] not "[1, 2, 3]"

PARAMETER VALIDATION RULES:
- scene_ids: Must be a JSON array of integers [1, 2, 3]
- scene_breakdown_json: Must be a JSON object, not a string
- model_name: Must be a string "nano-banana"
- aspect_ratio: Must be a string "9:16" or "1:1" or "16:9"

MERGE_VIDEOS_TOOL FORMATTING:
- video_clips: Must be a JSON array of objects, NOT a string
- Each video clip object must have: scene_id, video_url, duration, transition_type
- Example: Pass video_clips as a proper list like [{{"scene_id": 1, "video_url": "https://...", "duration": 5.0, "transition_type": "fade"}}]
- NEVER pass video_clips as a string - it must be an actual JSON array

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: {{"parameter_name": "parameter_value", "another_param": "value"}}
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


# Scene breakdown prompts
SCENE_BREAKDOWN_WITH_IMAGES = """Create a complete scene breakdown for this video request: {user_prompt}

Available images analysis:
{image_analysis}

CRITICAL CONSTRAINTS:
1. Each scene duration: 5 or 10 seconds EXACTLY (never longer or lesser then this - technical limitation)
2. Character consistency: Same face, body structure across ALL scenes unless user wants to change it
3. Garment consistency: Exact same garment style, pattern, design in ALL scenes
4. PDP images are reference only: Change backgrounds, poses, lighting freely, consider the PDP images as reference only

BACKGROUND ENHANCEMENT RULES:
- If source images have flat, plain, or minimal backgrounds, REPLACE with thematically appropriate realistic backgrounds
- For Diwali/Festival themes: Generate realistic Diwali settings with diyas, rangoli patterns, marigold decorations, traditional architecture
- For Wedding themes: Create elegant wedding venues with flowers, mandap decorations, traditional elements
- For Fashion/Lifestyle: Design modern, stylish environments that complement the garment and mood
- For Product showcase: Create professional studio or lifestyle settings that enhance the product appeal

INDEPENDENT API CALL REQUIREMENTS:
- Each scene will be generated via separate, independent API calls
- NO scene can reference another scene ("previous scene", "continuing from", "as seen before")
- Each image_edit_prompt and video_animation_prompt MUST be completely self-contained
- Include FULL character description, garment details, background, lighting, and mood in EVERY scene
- Repeat all necessary visual details in each scene as if it's the only scene being generated

Create a structured breakdown with:
- total_scenes: Number of scenes (2-5)
- estimated_duration: MUST BE EXACTLY {target_duration} seconds (user's requirement)
- aspect_ratio: MUST BE EXACTLY "{aspect_ratio}" (user's requirement)
- scenes: Array of scene objects

Each scene must have:
- scene_id: Sequential number starting from 1
- source_image_index: Which image to use (0-based, max {max_image_index})
- image_edit_prompt: DETAILED, SELF-CONTAINED prompt including:
  * Complete character description (face, body, pose)
  * Full garment description (style, color, pattern, fabric details)
  * Detailed background description (replace flat backgrounds with thematic ones)
  * Lighting and mood specifications
  * "Maintain character consistency and keep same garment style" instruction
- video_animation_prompt: DETAILED camera movements, effects, and visual elements (5-10 sec max)
- duration: Scene length (5 or 10 seconds only)
- transition_type: "fade", "cut", or "dissolve"

PROMPT WRITING GUIDELINES:
- Write each prompt as if generating a complete standalone image/video
- Include specific visual details, not general descriptions
- Describe lighting, composition, and atmosphere explicitly
- Mention specific background elements and decorations
- Avoid vague terms like "elegant" or "beautiful" - be specific about what makes it elegant

IMAGE QUALITY & COMPOSITION REQUIREMENTS:
- Ensure seamless blending between subject and background with consistent lighting conditions
- Match shadow direction, intensity, and color temperature throughout the scene
- Avoid harsh lighting transitions or unnatural shadow placement
- Use medium shots and full-body shots unless close-ups are specifically requested
- Maintain video-appropriate framing that allows for camera movement and animation
- Avoid extreme close-ups that limit video animation possibilities
- Ensure proper depth and perspective for realistic image composition

IMPORTANT: The aspect_ratio and estimated_duration MUST match user requirements exactly!"""


SCENE_BREAKDOWN_NO_IMAGES = """Create a complete scene breakdown for this video request: {user_prompt}

CRITICAL CONSTRAINTS:
1. Each scene duration: 5 or 10 seconds MAXIMUM (never longer - technical limitation)
2. Character consistency: Same face, body structure across ALL scenes
3. Garment consistency: Exact same garment style, pattern, design in ALL scenes
4. Be creative with backgrounds, poses, lighting while maintaining consistency

BACKGROUND ENHANCEMENT RULES:
- Create thematically appropriate realistic backgrounds for each scene
- For Diwali/Festival themes: Generate realistic Diwali settings with diyas, rangoli patterns, marigold decorations, traditional architecture
- For Wedding themes: Create elegant wedding venues with flowers, mandap decorations, traditional elements
- For Fashion/Lifestyle: Design modern, stylish environments that complement the garment and mood
- For Product showcase: Create professional studio or lifestyle settings that enhance the product appeal

INDEPENDENT API CALL REQUIREMENTS:
- Each scene will be generated via separate, independent API calls
- NO scene can reference another scene ("previous scene", "continuing from", "as seen before")
- Each image_edit_prompt and video_animation_prompt MUST be completely self-contained
- Include FULL character description, garment details, background, lighting, and mood in EVERY scene
- Repeat all necessary visual details in each scene as if it's the only scene being generated

Create a structured breakdown with:
- total_scenes: Number of scenes (2-5)
- estimated_duration: MUST BE EXACTLY {target_duration} seconds (user's requirement)
- aspect_ratio: MUST BE EXACTLY "{aspect_ratio}" (user's requirement)
- scenes: Array of scene objects

Each scene must have:
- scene_id: Sequential number starting from 1
- source_image_index: Suggest image index (0, 1, 2, etc.)
- image_edit_prompt: DETAILED, SELF-CONTAINED prompt including:
  * Complete character description (face, body, pose)
  * Full garment description (style, color, pattern, fabric details)
  * Detailed background description with thematic elements
  * Lighting and mood specifications
  * "Maintain character consistency and keep same garment style" instruction
- video_animation_prompt: DETAILED camera movements, effects, and visual elements (5-10 sec max)
- duration: Scene length (5 or 10 seconds only)
- transition_type: "fade", "cut", or "dissolve"

PROMPT WRITING GUIDELINES:
- Write each prompt as if generating a complete standalone image/video
- Include specific visual details, not general descriptions
- Describe lighting, composition, and atmosphere explicitly
- Mention specific background elements and decorations
- Avoid vague terms like "elegant" or "beautiful" - be specific about what makes it elegant

IMAGE QUALITY & COMPOSITION REQUIREMENTS:
- Ensure seamless blending between subject and background with consistent lighting conditions
- Match shadow direction, intensity, and color temperature throughout the scene
- Avoid harsh lighting transitions or unnatural shadow placement
- Use medium shots and full-body shots unless close-ups are specifically requested
- Maintain video-appropriate framing that allows for camera movement and animation
- Avoid extreme close-ups that limit video animation possibilities
- Ensure proper depth and perspective for realistic image composition

Focus on creating a professional, engaging video concept with strict consistency.

IMPORTANT: The aspect_ratio and estimated_duration MUST match user requirements exactly!"""


# Image analysis prompt
IMAGE_ANALYSIS_PROMPT = """Analyze the provided images for creating a video about: {user_prompt}

For each image, provide a detailed analysis focusing on:

CHARACTER ANALYSIS:
- Face features, expression, and overall appearance
- Body structure, pose, and positioning
- Hair style, makeup, and any accessories

GARMENT ANALYSIS:
- Exact garment type, style, and design
- Color, pattern, fabric texture, and material
- Specific details like embroidery, lace, prints, or embellishments
- Fit, drape, and how the garment falls on the body

BACKGROUND ASSESSMENT:
- Background type: Is it flat/plain, minimal, or detailed?
- Background color and texture (solid color, gradient, textured wall, etc.)
- If plain/flat background: Note that it needs thematic replacement
- If detailed background: Describe existing elements and setting
- Lighting setup and mood of the current background

COMPOSITION & TECHNICAL:
- Image composition and framing (close-up, medium shot, full-body, etc.)
- Lighting quality, direction, and mood (assess for consistency and blending)
- Shadow placement, intensity, and color temperature
- Overall color palette and visual style
- Depth and perspective quality
- Any text, branding, or graphic elements visible
- Assessment of how well subject blends with background

THEMATIC ENHANCEMENT SUGGESTIONS:
Based on the video concept "{user_prompt}", suggest:
- What type of background would enhance this image for the video theme
- Specific thematic elements that should be added (e.g., for Diwali: diyas, rangoli, marigolds)
- Lighting adjustments needed to match the video mood
- How this image could be transformed while maintaining character and garment consistency

IMPORTANT: Pay special attention to identifying flat/plain backgrounds that need thematic replacement and provide specific suggestions for enhancement.

Format your response as:
Image 0: [Complete analysis covering all above points]
Image 1: [Complete analysis covering all above points]
etc."""


# Enhanced input preparation template
ENHANCED_INPUT_TEMPLATE = """User Request: {user_input}

{image_info}

USER CONFIGURATION:
- Aspect Ratio: {aspect_ratio}
- Duration: {duration}
- LLM Model: {llm_model}
- Image Model: {image_model}
- Video Model: {video_model}

IMPORTANT: Use these exact configuration values when calling tools!"""
