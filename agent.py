"""
Pixora Video Agent - ReACT Architecture Implementation
"""
import os
from typing import List, Dict, Any, Optional
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv

# Import our custom tools
from tools.scene_breakdown import scene_breakdown_tool
from tools.image_generation import generate_scene_images_tool, regenerate_scene_images_tool
from tools.video_generation import generate_scene_videos_tool, regenerate_scene_videos_tool
from tools.video_merging import merge_videos_tool, regenerate_and_merge_videos_tool
from tools.pdp_image_extraction import extract_pdp_images_tool

# Import prompts
from config.prompts import AGENT_SYSTEM_PROMPT, ENHANCED_INPUT_TEMPLATE

# Load environment variables
load_dotenv()


class PixoraVideoAgent:
    """
    Main ReACT agent for video generation pipeline
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7, verbose: bool = True, session_key: str = "default"):
        """
        Initialize the Pixora Video Agent
        
        Args:
            model_name: OpenAI model to use for reasoning
            temperature: Temperature for LLM responses
            verbose: Whether to show detailed reasoning steps
            session_key: Unique key for this session's memory
        """
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.session_key = session_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        
        # Initialize Streamlit-specific chat message history
        self.chat_history = StreamlitChatMessageHistory(key=f"chat_messages_{session_key}")
        
        # Initialize memory for conversation history with Streamlit integration
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=self.chat_history
        )
        
        # Define available tools
        self.tools = [
            extract_pdp_images_tool,
            scene_breakdown_tool,
            generate_scene_images_tool,
            regenerate_scene_images_tool,
            generate_scene_videos_tool,
            regenerate_scene_videos_tool,
            merge_videos_tool,
            regenerate_and_merge_videos_tool,
        ]
        
        # Create the ReACT prompt template
        self.prompt = self._create_react_prompt()
        
        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=verbose,
            handle_parsing_errors="Check your output and make sure to follow the format! Always end with either Action: or Final Answer:",
            max_iterations=15,  # Reduced to prevent infinite loops
            max_execution_time=1000,  # 5 minutes timeout
            early_stopping_method="generate"  # Stop early if possible
        )
    
    def _create_react_prompt(self) -> PromptTemplate:
        """
        Create a custom ReACT prompt template for video generation
        """
        template = AGENT_SYSTEM_PROMPT

        return PromptTemplate(
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            },
            template=template
        )
    
    def process_request(self, user_input: str, image_paths: Optional[List[str]] = None, 
                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user request for video generation
        
        Args:
            user_input: User's video generation request
            image_paths: List of uploaded image file paths
            config: Configuration options (aspect_ratio, duration, etc.)
        
        Returns:
            Dictionary containing the agent's response and any generated content
        """
        
        # Store image paths and config in environment variables so tools can access them
        if image_paths:
            os.environ["PIXORA_IMAGE_PATHS"] = str(image_paths)
        else:
            os.environ.pop("PIXORA_IMAGE_PATHS", None)
            
        if config:
            os.environ["PIXORA_ASPECT_RATIO"] = config.get('aspect_ratio', '16:9')
            os.environ["PIXORA_DURATION"] = config.get('duration', '30sec')
            os.environ["PIXORA_IMAGE_MODEL"] = config.get('image_model', 'nano-banana')
            os.environ["PIXORA_VIDEO_MODEL"] = config.get('video_model', 'kling-v2')
        else:
            os.environ.pop("PIXORA_ASPECT_RATIO", None)
            os.environ.pop("PIXORA_DURATION", None)
            os.environ.pop("PIXORA_IMAGE_MODEL", None)
            os.environ.pop("PIXORA_VIDEO_MODEL", None)
        
        # Prepare the enhanced input with context
        enhanced_input = self._prepare_enhanced_input(user_input, image_paths, config)
        
        try:
            # Add user message to chat history
            self.chat_history.add_user_message(user_input)
            
            # Execute the agent
            result = self.agent_executor.invoke({
                "input": enhanced_input
            })
            
            # Get the agent's response
            agent_response = result.get("output", "")
            
            # Add agent response to chat history
            self.chat_history.add_ai_message(agent_response)
            
            return {
                "success": True,
                "response": agent_response,
                "chat_history": self.chat_history.messages,
                "intermediate_steps": result.get("intermediate_steps", [])
            }
            
        except Exception as e:
            error_response = f"I encountered an error while processing your request: {str(e)}"
            
            # Still add the error response to chat history
            self.chat_history.add_ai_message(error_response)
            
            return {
                "success": False,
                "error": str(e),
                "response": error_response,
                "chat_history": self.chat_history.messages
            }
        finally:
            # Clean up environment variables
            for key in ["PIXORA_IMAGE_PATHS", "PIXORA_ASPECT_RATIO", "PIXORA_DURATION", "PIXORA_IMAGE_MODEL", "PIXORA_VIDEO_MODEL"]:
                os.environ.pop(key, None)
    
    def _prepare_enhanced_input(self, user_input: str, image_paths: Optional[List[str]] = None,
                               config: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance user input with additional context using template
        """
        # Prepare image info
        image_info = ""
        if image_paths:
            image_info = f"Uploaded Images: {len(image_paths)} images provided\nImage paths: {image_paths}"
        
        # Use template for consistent formatting
        return ENHANCED_INPUT_TEMPLATE.format(
            user_input=user_input,
            image_info=image_info,
            aspect_ratio=config.get('aspect_ratio', '16:9') if config else '16:9',
            duration=config.get('duration', '30sec') if config else '30sec',
            llm_model=config.get('llm_model', 'GPT-4.0') if config else 'GPT-4.0',
            image_model=config.get('image_model', 'Kontext') if config else 'Kontext',
            video_model=config.get('video_model', 'Kling 1.6') if config else 'Kling 1.6'
        )
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get formatted conversation history
        """
        history = []
        for message in self.chat_history.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.chat_history.clear()
        self.memory.clear()
    
    def add_tool(self, tool):
        """Add a new tool to the agent"""
        self.tools.append(tool)
        # Recreate agent with updated tools
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=300
        )


def create_pixora_agent(config: Optional[Dict[str, Any]] = None) -> PixoraVideoAgent:
    """
    Factory function to create a Pixora Video Agent
    """
    default_config = {
        "model_name": os.getenv("DEFAULT_LLM_MODEL", "gpt-4o"),
        "temperature": 0.7,
        "verbose": os.getenv("VERBOSE_LOGGING", "true").lower() == "true"
    }
    
    if config:
        default_config.update(config)
    
    return PixoraVideoAgent(**default_config)


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = create_pixora_agent()
    
    # Test with sample request
    sample_request = "Create a 30-second Diwali advertisement video using my product images with festive lighting and celebration themes"
    sample_images = ["image1.jpg", "image2.jpg"]
    sample_config = {
        "aspect_ratio": "9:16",
        "duration": "30sec",
        "llm_model": "GPT-4.0"
    }
    
    result = agent.process_request(
        user_input=sample_request,
        image_paths=sample_images,
        config=sample_config
    )
    
    print("Agent Result:")
    print(f"Success: {result['success']}")
    print(f"Response: {result['response']}")
    
    if not result['success']:
        print(f"Error: {result.get('error', 'Unknown error')}")
