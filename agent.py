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
from dotenv import load_dotenv

# Import our custom tools
from tools.scene_breakdown import scene_breakdown_tool

# Load environment variables
load_dotenv()


class PixoraVideoAgent:
    """
    Main ReACT agent for video generation pipeline
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7, verbose: bool = True):
        """
        Initialize the Pixora Video Agent
        
        Args:
            model_name: OpenAI model to use for reasoning
            temperature: Temperature for LLM responses
            verbose: Whether to show detailed reasoning steps
        """
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Define available tools
        self.tools = [
            scene_breakdown_tool,
            # Additional tools will be added here:
            # - image_generation_tool
            # - video_animation_tool  
            # - video_merging_tool
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
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=300  # 5 minutes timeout
        )
    
    def _create_react_prompt(self) -> PromptTemplate:
        """
        Create a custom ReACT prompt template for video generation
        """
        template = """You are Pixora, an AI video generation agent specializing in creating engaging marketing videos from product images.

You have access to the following tools:
{tools}

Your expertise includes:
- Analyzing product images and understanding their marketing potential
- Creating compelling scene breakdowns for video narratives
- Coordinating image enhancement and video animation
- Ensuring brand consistency and visual appeal

WORKFLOW APPROACH:
1. First understand the user's request and analyze any provided images
2. Create a structured scene breakdown using the scene_breakdown_tool with proper parameters:
   - user_prompt: The user's video request
   - image_paths: List of uploaded image paths (or null if none provided)
   - aspect_ratio: From configuration (e.g., "9:16", "1:1")
   - duration_preference: From configuration (e.g., "30sec", "1min")
3. Present the plan to the user for approval before proceeding
4. Execute image enhancement and video generation (tools to be implemented)
5. Provide final video with clear delivery information

COMMUNICATION STYLE:
- Be professional yet friendly and creative
- Explain your reasoning clearly at each step
- Ask for user confirmation before major processing steps
- Provide progress updates during long operations

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

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
        
        # Prepare the enhanced input with context
        enhanced_input = self._prepare_enhanced_input(user_input, image_paths, config)
        
        try:
            # Execute the agent
            result = self.agent_executor.invoke({
                "input": enhanced_input
            })
            
            return {
                "success": True,
                "response": result.get("output", ""),
                "chat_history": self.memory.chat_memory.messages,
                "intermediate_steps": result.get("intermediate_steps", [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error while processing your request: {str(e)}",
                "chat_history": self.memory.chat_memory.messages
            }
    
    def _prepare_enhanced_input(self, user_input: str, image_paths: Optional[List[str]] = None,
                               config: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance user input with additional context
        """
        enhanced_parts = [f"User Request: {user_input}"]
        
        if image_paths:
            enhanced_parts.append(f"Uploaded Images: {len(image_paths)} images provided")
            enhanced_parts.append(f"Image paths: {image_paths}")
        
        if config:
            config_str = ", ".join([f"{k}: {v}" for k, v in config.items()])
            enhanced_parts.append(f"Configuration: {config_str}")
        
        return "\n\n".join(enhanced_parts)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get formatted conversation history
        """
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history
    
    def clear_memory(self):
        """Clear conversation memory"""
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
