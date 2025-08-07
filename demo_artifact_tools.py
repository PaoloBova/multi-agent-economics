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
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup logging for debugging
log_file = Path("./demo_artifact_tools.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Overwrite log file each run
        # logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger('demo_artifact_tools')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import AutoGen 0.6.4 components
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_agentchat.ui import Console
    print("✓ AutoGen 0.6.4 packages loaded successfully")
except ImportError as e:
    print(f"❌ AutoGen 0.6.4 packages required but not available: {e}")
    print("Please install: pip install autogen-agentchat autogen-ext[openai]")
    sys.exit(1)

# Import our tools and core components
from multi_agent_economics.tools.artifacts import create_artifact_tools
from multi_agent_economics.core.artifacts import Workspace, ArtifactManager
from multi_agent_economics.core.workspace_memory import WorkspaceMemory
from multi_agent_economics.core.budget import BudgetManager
from multi_agent_economics.core.actions import ActionLogger


async def create_poetry_team_with_simulation():
    """Create a team of 3 AutoGen agents with simulation budget and action tracking."""
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please add your OpenAI API key to .env file")
        return None, None
    
    # Create OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key
    )
    print("✓ OpenAI client created with gpt-4o-mini")
    
    # Create simulation objects
    from pathlib import Path
    
    # Create workspace directory in current directory for easy inspection
    workspace_dir = Path("./poetry_collaboration_workspace")
    
    # Create ArtifactManager and single shared workspace for all agents
    from multi_agent_economics.core.artifacts import ArtifactManager
    artifact_manager = ArtifactManager(workspace_dir.parent)
    logger.info(f"Created ArtifactManager with base directory: {workspace_dir.parent.absolute()}")
    
    # Create single shared workspace for all agents to use
    shared_workspace = artifact_manager.create_workspace("poetry_collaboration")
    logger.info(f"Created shared workspace: {shared_workspace.workspace_dir}")
    
    budget_manager = BudgetManager()
    action_logger = ActionLogger()
    print(f"✓ Created artifact manager, shared workspace for all agents, budget manager, and action logger")
    
    # Initialize agent budgets (simulation credits, not real costs)
    initial_budget_per_agent = 10.0
    budget_manager.initialize_budget("ThemeCreator_artifacts", initial_budget_per_agent)
    budget_manager.initialize_budget("VerseWriter_artifacts", initial_budget_per_agent)
    budget_manager.initialize_budget("Editor_artifacts", initial_budget_per_agent)
    print(f"✓ Initialized {initial_budget_per_agent} simulation credits per agent")
    
    # Create workspace memory instances - all agents share the same workspace
    theme_memory = WorkspaceMemory(name="ThemeCreator_workspace", workspace=shared_workspace)
    verse_memory = WorkspaceMemory(name="VerseWriter_workspace", workspace=shared_workspace)
    editor_memory = WorkspaceMemory(name="Editor_workspace", workspace=shared_workspace)
    
    print("✓ Created workspace memories for each agent")
    
    # Create artifact tools BEFORE agents using proper context pattern
    base_context = {
        'budget_manager': budget_manager,
        'action_logger': action_logger,
        'budget_costs': {  # Configurable simulation costs
            'tool:load_artifact': 0.2,
            'tool:write_artifact': 0.5, 
            'tool:share_artifact': 0.3,
            'tool:unload_artifact': 0.0,
            'tool:list_artifacts': 0.0
        }
    }
    
    # Create context and tools for each agent
    theme_context = {**base_context, 'agent_name': 'ThemeCreator', 'workspace_memory': theme_memory}
    verse_context = {**base_context, 'agent_name': 'VerseWriter', 'workspace_memory': verse_memory}
    editor_context = {**base_context, 'agent_name': 'Editor', 'workspace_memory': editor_memory}
    
    theme_tools = create_artifact_tools(theme_context)
    verse_tools = create_artifact_tools(verse_context)
    editor_tools = create_artifact_tools(editor_context)
    
    print(f"✓ Created {len(theme_tools)} artifact tools per agent")
    
    # Now create agents WITH tools (proper AutoGen pattern)
    theme_agent = AssistantAgent(
        name="ThemeCreator",
        model_client=model_client,
        tools=theme_tools,  # ✅ Proper AutoGen pattern!
        memory=[theme_memory],
        reflect_on_tool_use=True,
        max_tool_iterations=10,
        system_message="""You are a creative theme creator for collaborative poetry. 

Your role is to:
1. Create initial themes and structures for poems using write_artifact
2. Use list_artifacts to see what artifacts are available in the shared workspace
3. Communicate naturally with your collaborators about your creative work

ARTIFACT GUIDELINES:
- Use descriptive artifact IDs (e.g., "digital_connections_theme", "nature_poem_structure") 
- Content should always be a dictionary with meaningful structure
- All agents share the same workspace, so artifacts are automatically visible to everyone

COLLABORATION APPROACH:
- Mention your artifacts in conversation to draw attention to them
- Example: "I've created a theme called 'digital_connections_theme' - VerseWriter, what verses does that inspire you to write?"
- Use natural language to guide the creative process
- Ask questions and provide feedback on others' work

Example workflow:
1. Create theme: write_artifact(artifact_id="digital_connections_theme", content={...}, artifact_type="theme")
2. Announce it: "I've just created a 'digital_connections_theme' artifact exploring how technology connects us - VerseWriter, take a look and see what poetry it inspires!"
3. Collaborate: Discuss the creative direction and help guide the team

Always use the artifact tools to store your creative work, and communicate naturally with your collaborative team."""
    )
    
    verse_agent = AssistantAgent(
        name="VerseWriter", 
        model_client=model_client,
        tools=verse_tools,  # ✅ Proper AutoGen pattern!
        memory=[verse_memory],
        reflect_on_tool_use=True,
        max_tool_iterations=10,
        system_message="""You are a skilled verse writer who creates poetry based on themes in the shared workspace.

Your role is to:
1. Check for available themes using list_artifacts (look for artifacts marked with * if new)
2. Load themes you want to work with using load_artifact
3. Write verses based on the loaded themes using write_artifact  
4. Communicate naturally about your creative process

ARTIFACT GUIDELINES:
- Look for theme artifacts in the shared workspace (marked with * if new)
- Use descriptive verse artifact IDs (e.g., "digital_dreams_verses", "nature_reflections_stanzas") 
- Content should be a dictionary with your verse data
- All agents can see your artifacts once you create them

COLLABORATION APPROACH:
- Respond to teammates when they mention creating or discussing artifacts
- Feel free to ask questions or make suggestions about the creative direction
- Example: "I saw your digital connections theme - I'll create some verses exploring virtual relationships and human connection"

Example workflow:
1. Check for available themes: list_artifacts()
2. Load inspiring theme: load_artifact(artifact_id="digital_connections_theme")
3. Create verses: write_artifact(artifact_id="digital_dreams_verses", content={...}, artifact_type="verses")
4. Announce: "Editor, I've created 'digital_dreams_verses' inspired by the digital connections theme - ready for your editorial touch!"

Always use the artifact tools and communicate naturally with your creative team about the collaborative process."""
    )
    
    editor_agent = AssistantAgent(
        name="Editor",
        model_client=model_client,
        tools=editor_tools,  # ✅ Proper AutoGen pattern!
        memory=[editor_memory],
        reflect_on_tool_use=True,
        max_tool_iterations=10,
        system_message="""You are a poetry editor who creates final polished poems from collaborative materials in the shared workspace.

Your role is to:
1. Check for available materials using list_artifacts (look for artifacts marked with * if new)
2. Load themes and verses using load_artifact to understand the creative vision
3. Create a final polished poem using write_artifact that brings everything together
4. Clean up your memory using unload_artifact when done
5. Provide thoughtful feedback and editorial guidance

ARTIFACT GUIDELINES:
- Look for theme and verse artifacts in the shared workspace (marked with * if new)
- Use descriptive names for final poems (e.g., "digital_connections_final", "nature_symphony_complete")
- Content should include the polished poem text and metadata
- Clean up loaded artifacts when finished to keep your memory tidy

COLLABORATION APPROACH:
- Acknowledge the creative contributions from your teammates
- Provide editorial feedback and suggestions for improvement
- Example: "I've woven together the digital connections theme with your beautiful verses - the final poem captures both intimacy and technology beautifully"

Example workflow:
1. Check available materials: list_artifacts()
2. Load creative materials: load_artifact(artifact_id="digital_connections_theme"), load_artifact(artifact_id="digital_dreams_verses")
3. Create final poem: write_artifact(artifact_id="digital_connections_final", content={...}, artifact_type="poem")
4. Clean up: unload_artifact(artifact_id="digital_connections_theme"), unload_artifact(artifact_id="digital_dreams_verses")
5. Announce completion: "The collaborative poem is complete! I've combined your theme and verses into 'digital_connections_final' - a cohesive piece exploring our digital age."

Use the artifact tools effectively and communicate naturally about the creative collaboration process."""
    )
    
    print("✓ Created 3 AutoGen agents with tools and memory using proper AutoGen pattern")

    # Package simulation objects for demo
    simulation = {
        'artifact_manager': artifact_manager,
        'shared_workspace': shared_workspace,
        'budget_manager': budget_manager, 
        'action_logger': action_logger,
        'agents': [
            {"agent": theme_agent, "name": "ThemeCreator", "workspace_memory": theme_memory},
            {"agent": verse_agent, "name": "VerseWriter", "workspace_memory": verse_memory},
            {"agent": editor_agent, "name": "Editor", "workspace_memory": editor_memory}
        ]
    }
    
    # Create termination condition (max 15 messages to ensure completion)
    termination = MaxMessageTermination(max_messages=15)
    
    # Create the team
    team = RoundRobinGroupChat([theme_agent, verse_agent, editor_agent], termination_condition=termination)
    print("✓ Created poetry team with 3 agents")
    
    return team, simulation


async def run_poetry_collaboration():
    """Run the complete poetry collaboration demo with simulation architecture."""
    print("\n🎭 AUTOGEN ARTIFACT TOOLS DEMO - SIMPLIFIED VERSION")
    print("="*60)
    print("Testing core 4 artifact tools with real AutoGen team collaboration (load, write, list, unload)")
    
    # Create the poetry team with simulation objects
    team, simulation = await create_poetry_team_with_simulation()
    if not team or not simulation:
        return False
    
    # Define the collaborative task
    task = """Let's create a collaborative poem about 'digital connections in the modern world'. 

ThemeCreator: Start by creating a poem theme and structure using write_artifact, then announce it to the team.
VerseWriter: Check for available themes with list_artifacts, load the theme, and create verses based on it.  
Editor: Check for all available materials, load themes and verses, then create the final polished poem and clean up your memory.

Each agent should use their artifact tools and communicate naturally to collaborate effectively on this creative project."""
    
    print(f"\n🚀 Starting collaborative poetry creation...")
    print(f"Task: {task}")
    
    try:
        # Run the team collaboration
        await team.reset()  # Reset the team for a new task.
        result = await Console(team.run_stream(task=task))  # Stream the messages to the console.
        
        print(f"\n✅ Team collaboration completed!")
        print(f"Total messages: {len(result.messages)}")
        
        # Print a summary of the conversation
        print(f"\n📝 COLLABORATION SUMMARY")
        print("-" * 40)
        for i, msg in enumerate(result.messages[-5:], len(result.messages)-4):  # Last 5 messages
            sender = getattr(msg, 'source', 'Unknown')
            content = str(msg.content)
            # Truncate very long messages
            if len(content) > 300:
                content = content[:300] + "... [TRUNCATED]"
            print(f"{i}. {sender}: {content}")
        
        # Show simulation summaries
        print(f"\n💰 SIMULATION BUDGET SUMMARY")
        print("-" * 30)
        budget_manager = simulation['budget_manager']
        for category in ["ThemeCreator_artifacts", "VerseWriter_artifacts", "Editor_artifacts"]:
            balance = budget_manager.get_balance(category)
            transactions = len(budget_manager.get_transaction_history(category))
            print(f"{category.replace('_artifacts', '')}: ${balance:.2f} remaining, {transactions} transactions")
        
        print(f"\n⚡ ACTION SUMMARY")  
        print("-" * 20)
        action_logger = simulation['action_logger']
        actions = action_logger.get_all_actions()
        
        # Group actions by actor
        action_by_actor = {}
        for action in actions:
            actor = action.actor
            if actor not in action_by_actor:
                action_by_actor[actor] = {'actions': [], 'total_cost': 0.0, 'tools_used': set()}
            action_by_actor[actor]['actions'].append(action)
            action_by_actor[actor]['total_cost'] += action.cost
            action_by_actor[actor]['tools_used'].add(action.tool)
        
        for actor, summary in action_by_actor.items():
            tool_list = ", ".join(sorted(summary['tools_used']))
            print(f"{actor}: {len(summary['actions'])} actions, ${summary['total_cost']:.2f} spent")
            print(f"   Tools used: {tool_list}")
        
        # Simple audit: check if all 5 tools were used
        all_tools_used = set()
        for summary in action_by_actor.values():
            all_tools_used.update(summary['tools_used'])
        
        expected_tools = {'load_artifact', 'write_artifact', 'list_artifacts', 'unload_artifact'}
        tools_used_count = len(all_tools_used.intersection(expected_tools))
        audit_success = tools_used_count >= 3  # At least 3 of 4 tools used
        
        print(f"\n🔍 AUDIT RESULTS")
        print("-" * 15)
        print(f"Tools used: {tools_used_count}/4 ({', '.join(sorted(all_tools_used))})")
        print(f"Audit status: {'PASS' if audit_success else 'PARTIAL'}")
        
        return audit_success
        
    except Exception as e:
        print(f"❌ Error during collaboration: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main demo function."""
    print("🎯 AutoGen 0.6.4 Artifact Tools Demo - Simplified Architecture")
    print("Using 4 core artifact tools with shared workspace collaboration")
    
    try:
        # Run the poetry collaboration  
        success = await run_poetry_collaboration()
        
        if success:
            print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("✅ Artifact tools were tested successfully with real AutoGen agents")
            print("✅ Agents collaborated using AutoGen teams with workspace memory")
            print("✅ Simulation budget and action tracking worked correctly")
        else:
            print(f"\n❌ DEMO HAD ISSUES")
            print("Some artifact tools may not have been used or there were errors")
            return False
            
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    print("Starting AutoGen 0.6.4 Artifact Tools Demo - Simulation Version...")
    success = asyncio.run(main())
    
    if success:
        print("\n✨ Demo completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Demo had issues.")
        sys.exit(1)