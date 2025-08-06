#!/usr/bin/env python3
"""
AutoGen Artifact Tools Demo

This script demonstrates all 5 artifact tools working with a real team of AutoGen 0.6.4 agents
collaborating on a poetry creation project. The goal is to verify that:

1. All 5 artifact tools work correctly with AutoGen agents
2. Agents can share artifacts between each other using the tools
3. Full audit trail is maintained of tool usage
4. Real AutoGen team workflow with proper agent collaboration

The demo creates 3 AutoGen agents in a RoundRobinGroupChat:
- Theme Creator: Creates initial poem theme and structure
- Verse Writer: Develops verses based on shared themes  
- Editor: Refines and finalizes the poem using all shared materials

Requires: AutoGen 0.6.4, OpenAI API key in .env file
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import AutoGen 0.6.4 components
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    print("✓ AutoGen 0.6.4 packages loaded successfully")
except ImportError as e:
    print(f"❌ AutoGen 0.6.4 packages required but not available: {e}")
    print("Please install: pip install autogen-agentchat autogen-ext[openai]")
    sys.exit(1)

# Import our artifact tools and infrastructure
from multi_agent_economics.tools.artifacts import create_artifact_tools_for_agent
from multi_agent_economics.core.actions import ActionLogger
from multi_agent_economics.core.budget import BudgetManager
from multi_agent_economics.core.workspace_memory import WorkspaceMemory


class AgentWithTools:
    """
    Wrapper to add our custom budget manager, action logger, and workspace to AutoGen agents.
    This allows our artifact tools to work with the AutoGen agents.
    """
    
    def __init__(self, autogen_agent, name: str):
        self.autogen_agent = autogen_agent
        self.name = name
        
        # Add our custom components needed by artifact tools
        self.budget_manager = BudgetManager()
        self.action_logger = ActionLogger() 
        self.workspace_memory = WorkspaceMemory()
        
        # Initialize budget for artifact tools
        self.budget_manager.set_agent_budget(f"{name}_artifacts", 10.0)
        
        print(f"✓ Enhanced agent {name} with artifact tool support")


async def create_artifact_tool_functions(agent_wrapper):
    """
    Create async functions that can be used as AutoGen tools.
    These wrap our artifact tools to work with AutoGen's tool system.
    """
    
    # Get the artifact tools for this agent
    tools = create_artifact_tools_for_agent(agent_wrapper)
    
    # Create wrapper functions that can be used as AutoGen tools
    async def write_artifact(artifact_id: str, content: dict, artifact_type: str = "analysis") -> str:
        """Write or update an artifact in the workspace."""
        write_tool = next(t for t in tools if t.name == "write_artifact")
        result = await write_tool.run_json({
            "artifact_id": artifact_id,
            "content": content,
            "artifact_type": artifact_type
        }, None)
        return f"Artifact '{artifact_id}' written successfully. Status: {result.get('status', 'unknown')}"
    
    async def load_artifact(artifact_id: str) -> str:
        """Load an artifact into memory for use in next prompt."""
        load_tool = next(t for t in tools if t.name == "load_artifact") 
        result = await load_tool.run_json({"artifact_id": artifact_id}, None)
        return f"Loaded artifact '{artifact_id}'. Status: {result.get('status', 'unknown')}. Content available in workspace memory."
    
    async def share_artifact(artifact_id: str, target_agent: str) -> str:
        """Share an artifact with another agent."""
        share_tool = next(t for t in tools if t.name == "share_artifact")
        result = await share_tool.run_json({
            "artifact_id": artifact_id,
            "target_agent": target_agent
        }, None)
        return f"Shared artifact '{artifact_id}' with {target_agent}. Status: {result.get('status', 'unknown')}"
    
    async def list_artifacts() -> str:
        """List all artifacts available in the workspace."""
        list_tool = next(t for t in tools if t.name == "list_artifacts")
        result = await list_tool.run_json({}, None)
        artifacts = result.get('artifacts', [])
        return f"Found {len(artifacts)} artifacts: {[a.get('id', 'unknown') for a in artifacts]}"
    
    async def unload_artifact(artifact_id: str) -> str:
        """Unload an artifact from memory."""
        unload_tool = next(t for t in tools if t.name == "unload_artifact")
        result = await unload_tool.run_json({"artifact_id": artifact_id}, None)
        return f"Unloaded artifact '{artifact_id}'. Status: {result.get('status', 'unknown')}"
    
    return [write_artifact, load_artifact, share_artifact, list_artifacts, unload_artifact]


async def create_poetry_team():
    """Create a team of 3 AutoGen agents with artifact tools for collaborative poetry creation."""
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please add your OpenAI API key to .env file")
        return None
    
    # Create OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key
    )
    print("✓ OpenAI client created with gpt-4o-mini")
    
    # Create agent wrappers with our custom components
    theme_wrapper = AgentWithTools(None, "ThemeCreator")
    verse_wrapper = AgentWithTools(None, "VerseWriter")
    editor_wrapper = AgentWithTools(None, "Editor")
    
    # Create artifact tool functions for each agent
    theme_tools = await create_artifact_tool_functions(theme_wrapper)
    verse_tools = await create_artifact_tool_functions(verse_wrapper)
    editor_tools = await create_artifact_tool_functions(editor_wrapper)
    
    # Create AutoGen AssistantAgents with artifact tools
    theme_agent = AssistantAgent(
        name="ThemeCreator",
        model_client=model_client,
        tools=theme_tools,
        system_message="""You are a creative theme creator for collaborative poetry. 
        
Your role is to:
1. Create initial themes and structures for poems using write_artifact
2. Share your themes with other agents using share_artifact
3. Use list_artifacts to track your work

Always use the artifact tools to store and share your creative work. 
Create a theme artifact called 'poem_theme' with your initial ideas."""
    )
    
    verse_agent = AssistantAgent(
        name="VerseWriter", 
        model_client=model_client,
        tools=verse_tools,
        system_message="""You are a skilled verse writer who creates poetry based on shared themes.

Your role is to:
1. Load shared themes using load_artifact
2. Write verses based on the loaded themes using write_artifact  
3. Share your verses with other agents using share_artifact
4. List artifacts to see available materials

Always load the 'poem_theme' artifact first, then create verses in a new artifact."""
    )
    
    editor_agent = AssistantAgent(
        name="Editor",
        model_client=model_client,
        tools=editor_tools,
        system_message="""You are a poetry editor who creates final polished poems from collaborative work.

Your role is to:
1. Load all available artifacts (theme and verses) using load_artifact
2. Create a final polished poem using write_artifact
3. Clean up by unloading artifacts using unload_artifact when done
4. List artifacts to track the creative process

Load both 'poem_theme' and any verse artifacts, then create a final 'completed_poem' artifact."""
    )
    
    # Update the wrappers with the actual AutoGen agents
    theme_wrapper.autogen_agent = theme_agent
    verse_wrapper.autogen_agent = verse_agent  
    editor_wrapper.autogen_agent = editor_agent
    
    # Create termination condition (max 20 messages to ensure completion)
    termination = MaxMessageTermination(max_messages=20)
    
    # Create the team
    team = RoundRobinGroupChat([theme_agent, verse_agent, editor_agent], termination_condition=termination)
    print("✓ Created poetry team with 3 agents")
    
    return team, [theme_wrapper, verse_wrapper, editor_wrapper]


class ToolUsageAuditor:
    """Audits which artifact tools were used by the team."""
    
    def __init__(self, agent_wrappers):
        self.agent_wrappers = agent_wrappers
        
    def audit_tool_usage(self):
        """Generate audit report of tool usage."""
        print("\n" + "="*60)
        print("🔍 ARTIFACT TOOLS AUDIT REPORT")  
        print("="*60)
        
        all_tools_used = set()
        total_actions = 0
        
        for wrapper in self.agent_wrappers:
            actions = wrapper.action_logger.internal_actions
            print(f"\n📊 Agent: {wrapper.name}")
            print("-" * 30)
            
            if not actions:
                print("   No tool usage recorded")
                continue
                
            agent_tools = set()
            for action in actions:
                if action.tool:
                    agent_tools.add(action.tool)
                    all_tools_used.add(action.tool)
                    total_actions += 1
                    print(f"   ⚡ {action.action} ({action.tool}) - Cost: {action.cost}")
                    
            print(f"   📈 Tools used: {len(agent_tools)}")
            print(f"   🔧 Tool list: {sorted(list(agent_tools))}")
        
        print(f"\n📈 SUMMARY")
        print("-" * 20)
        print(f"Total agents: {len(self.agent_wrappers)}")
        print(f"Total tool actions: {total_actions}")
        print(f"Unique tools used: {len(all_tools_used)}")
        print(f"Tools used: {sorted(list(all_tools_used))}")
        
        # Check if all 5 artifact tools were used
        expected_tools = {'load_artifact', 'unload_artifact', 'write_artifact', 'share_artifact', 'list_artifacts'}
        missing_tools = expected_tools - all_tools_used
        
        if not missing_tools:
            print("✅ SUCCESS: All 5 artifact tools were tested!")
            return True
        else:
            print(f"❌ MISSING: The following tools were not used: {missing_tools}")
            return False


async def run_poetry_collaboration():
    """Run the complete poetry collaboration demo."""
    print("\n🎭 AUTOGEN ARTIFACT TOOLS DEMO")
    print("="*50)
    print("Creating a collaborative poem using AutoGen agents and artifact tools")
    
    # Create the poetry team
    team, agent_wrappers = await create_poetry_team()
    if not team:
        return False
    
    # Define the collaborative task
    task = """Let's create a collaborative poem about 'digital connections in the modern world'. 

ThemeCreator: Start by creating a theme and structure using your artifact tools.
VerseWriter: Load the theme and write verses based on it.
Editor: Load all materials and create the final polished poem.

Each agent should use their artifact tools appropriately and share work with the next agent."""
    
    print(f"\n🚀 Starting collaborative poetry creation...")
    print(f"Task: {task}")
    
    try:
        # Run the team collaboration
        result = await team.run(task=task)
        
        print(f"\n✅ Team collaboration completed!")
        print(f"Total messages: {len(result.messages)}")
        
        # Print the conversation
        print(f"\n📝 COLLABORATION TRANSCRIPT")
        print("-" * 40)
        for i, msg in enumerate(result.messages, 1):
            sender = getattr(msg, 'source', 'Unknown')
            content = str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content)
            print(f"{i}. {sender}: {content}")
        
        # Audit tool usage after collaboration
        auditor = ToolUsageAuditor(agent_wrappers)
        audit_success = auditor.audit_tool_usage()
        
        return audit_success
        
    except Exception as e:
        print(f"❌ Error during collaboration: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main demo function."""
    print("🎯 AutoGen 0.6.4 Artifact Tools Demo")
    print("Real team of agents using artifact tools for collaboration")
    
    try:
        # Run the poetry collaboration
        collaboration_success = await run_poetry_collaboration()
        
        if collaboration_success:
            print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("Agents collaborated using AutoGen teams and artifact tools.")
        else:
            print(f"\n❌ DEMO HAD ISSUES")
            print("Check the error messages above.")
            return False
            
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    print("Starting AutoGen 0.6.4 Artifact Tools Demo...")
    success = asyncio.run(main())
    
    if success:
        print("\n✨ Demo completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Demo had issues.")
        sys.exit(1)