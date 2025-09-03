import streamlit as st
from datetime import datetime
import os
import io
from PIL import Image
import tempfile
from agent import create_pixora_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Pixora Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for chat history and agent
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "processing" not in st.session_state:
    st.session_state.processing = False

# Sidebar with configuration section
with st.sidebar:
    # Add logo at the top
    st.image("logo.png", width=200)
    
    st.markdown("### Configuration")
    
    # Aspect ratio dropdown
    aspect_ratio = st.selectbox(
        "Aspect Ratio",
        ["1:1", "9:16"],
        index=0,
        help="Choose the aspect ratio for generated content"
    )
    
    # Duration dropdown
    duration = st.selectbox(
        "Duration",
        ["10sec", "30sec", "1min"],
        index=0,
        help="Choose the duration for video content"
    )
    
    # LLM selection dropdown
    llm_option = st.selectbox(
        "LLM",
        ["GPT-4.0", "GPT-5 Thinking", "GPT-4.5", "Claude 4 Sonnet", "Gemini 2.5 Pro"],
        index=0,
        help="Choose the AI language model"
    )
    
    # Image model dropdown
    image_option = st.selectbox(
        "Image",
        ["Kontext", "Nano Banana 🍌"],
        index=0,
        help="Choose the image generation model"
    )
    
    # Video model dropdown
    video_option = st.selectbox(
        "Video",
        ["Kling 1.6", "Kling 2.1", "Veo 3"],
        index=0,
        help="Choose the video generation model"
    )
    
    # st.markdown("---")
    # st.markdown("### 🔑 API Status")
    
    # # Check API key status
    # openai_key = os.getenv("OPENAI_API_KEY")
    # replicate_key = os.getenv("REPLICATE_API_TOKEN")
    
    # if openai_key:
    #     st.success("✅ OpenAI API Key")
    # else:
    #     st.error("❌ OpenAI API Key Missing")
    #     st.markdown("*Set OPENAI_API_KEY in your environment*")
    
    # if replicate_key:
    #     st.success("✅ Replicate API Token")
    # else:
    #     st.error("❌ Replicate API Token Missing")
    #     st.markdown("*Set REPLICATE_API_TOKEN in your environment*")

# Main chat interface

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display attached files if any
            if "files" in message and message["files"]:
                st.markdown("**Attached files:**")
                for file_info in message["files"]:
                    st.markdown(f"📎 {file_info['name']} ({file_info['size']} bytes)")

# Chat input with built-in file attachment support
prompt = st.chat_input(
    "Type a message and/or attach files…",
    accept_file="multiple",
    file_type=["pdf", "txt", "csv", "png", "jpg", "jpeg", "doc", "docx", "gif", "xlsx"]
)



if prompt:
    user_text = prompt.text or ""
    user_files = prompt.files or []
    
    # Prepare file information for storage
    file_info = []
    if user_files:
        for file in user_files:
            file_info.append({
                "name": file.name,
                "size": len(file.getbuffer()),
                "type": file.type or 'unknown'
            })
    
    # Add user message to chat history
    user_message = {
        "role": "user", 
        "content": user_text,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "files": file_info
    }
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        if user_text:
            st.markdown(user_text)
        if user_files:
            st.markdown("**Attached files:**")
            for file in user_files:
                if file.type and file.type.startswith("image/"):
                    st.image(Image.open(io.BytesIO(file.getbuffer())))
                else:
                    st.write(f"📎 {file.name} · {file.type or 'unknown'} · {len(file.getbuffer())} bytes")
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Simulate AI response (replace with actual AI integration)
            # Initialize agent if not exists
            agent_initialized = True
            if st.session_state.agent is None:
                try:
                    agent_config = {
                        "model_name": "gpt-4o",  # Map from UI selection if needed
                        "temperature": 0.7,
                        "verbose": True
                    }
                    st.session_state.agent = create_pixora_agent(agent_config)
                    st.success("✅ Pixora AI Agent initialized!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize agent: {str(e)}")
                    st.markdown("*Please check your API keys in the environment variables.*")
                    agent_initialized = False
            
            if agent_initialized:
                # Process files and save temporarily
                temp_image_paths = []
                if user_files:
                    for file in user_files:
                        if file.type and file.type.startswith("image/"):
                            # Save uploaded image to temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.type.split('/')[-1]}") as tmp_file:
                                tmp_file.write(file.getbuffer())
                                temp_image_paths.append(tmp_file.name)
                
                # Prepare configuration from UI
                config = {
                    "aspect_ratio": aspect_ratio,
                    "duration": duration,
                    "llm_model": llm_option,
                    "image_model": image_option,
                    "video_model": video_option
                }
                
                # Process request with agent
                st.session_state.processing = True
                
                try:
                    with st.spinner("🤖 Pixora AI is thinking and planning your video..."):
                        result = st.session_state.agent.process_request(
                            user_input=user_text,
                            image_paths=temp_image_paths if temp_image_paths else None,
                            config=config
                        )
                    
                    if result["success"]:
                        response_content = result["response"]
                        
                        # Show intermediate steps if available
                        if result.get("intermediate_steps"):
                            with st.expander("🔍 View AI Reasoning Steps", expanded=False):
                                for i, (action, observation) in enumerate(result["intermediate_steps"]):
                                    st.markdown(f"**Step {i+1}:**")
                                    st.markdown(f"*Action:* {action}")
                                    st.markdown(f"*Observation:* {observation}")
                                    st.markdown("---")
                    else:
                        response_content = f"❌ **Error:** {result['response']}"
                        if result.get("error"):
                            st.error(f"Technical details: {result['error']}")
                    
                    st.markdown(response_content)
                    
                except Exception as e:
                    st.error(f"❌ Agent processing failed: {str(e)}")
                    response_content = "I encountered an error while processing your request. Please try again or check your configuration."
                    st.markdown(response_content)
                
                finally:
                    # Clean up temporary files
                    for temp_path in temp_image_paths:
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                    
                    st.session_state.processing = False
            else:
                response_content = "❌ Agent initialization failed. Please check your API keys and try again."
                st.markdown(response_content)
    
    # Add assistant response to chat history
    assistant_message = {
        "role": "assistant",
        "content": response_content,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.messages.append(assistant_message)
