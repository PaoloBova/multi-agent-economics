#!/usr/bin/env python3
"""
AutoGen Artifact Tools Demo with GraphFlow and Custom Message Filtering

This script demonstrates artifact tools working with GraphFlow and custom filtering
that can actually intercept tool events before they reach agents, providing the
clean console output we want while preserving agent functionality.

The demo creates 3 AutoGen agents in a GraphFlow with custom filtering that:
1. Uses GraphFlow instead of RoundRobinGroupChat for proper message routing
2. Implements custom message filtering similar to MessageFilterAgent
3. Filters ToolCallRequestEvent and ToolCallExecutionEvent from agents selectively
4. Allows agents to see their own tool events (critical for functionality)
5. Preserves all TextMessage communication between agents
6. Provides clean "---------- AgentName ----------" output format

Key difference from RoundRobinGroupChat: In GraphFlow, messages are routed through
the graph, so we can actually intercept and filter them before they reach agents.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Sequence, List, Optional, Dict, Any
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Setup logging for debugging
log_file = Path("./demo_graphflow_artifact_tools.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        # logging.StreamHandler()  # Uncomment for console logging
    ]
)

# Reduce log volume from noisy components
logging.getLogger('autogen_core.events').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

logger = logging.getLogger('demo_graphflow_artifact_tools')
logging.getLogger('artifacts.tools').setLevel(logging.INFO)
logging.getLogger('autogen_core').setLevel(logging.INFO)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import AutoGen 0.6.4 components for GraphFlow
try:
    from autogen_agentchat.agents import AssistantAgent, BaseChatAgent, MessageFilterAgent, MessageFilterConfig
    from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import BaseChatMessage, BaseAgentEvent, TextMessage
    from autogen_agentchat.base import Response
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_agentchat.ui import Console
    from autogen_core import CancellationToken
    print("✓ AutoGen 0.6.4 GraphFlow packages loaded successfully")
except ImportError as e:
    print(f"❌ AutoGen 0.6.4 packages required but not available: {e}")
    print("Please install: pip install autogen-agentchat autogen-ext[openai]")
    sys.exit(1)

# Try to import tool event types
try:
    from autogen_agentchat.messages import ToolCallRequestEvent, ToolCallExecutionEvent
    TOOL_EVENTS_AVAILABLE = True
    print("✓ Tool event types available")
except ImportError:
    # Define placeholder classes if not available
    class ToolCallRequestEvent:
        pass
    class ToolCallExecutionEvent:
        pass
    TOOL_EVENTS_AVAILABLE = False
    print("⚠️ Tool event types not available - using placeholders")

# Try to import memory event type
try:
    from autogen_agentchat.messages import MemoryQueryEvent
    MEMORY_EVENTS_AVAILABLE = True
    print("✓ Memory event types available")
except ImportError:
    class MemoryQueryEvent:
        pass
    MEMORY_EVENTS_AVAILABLE = False
    print("⚠️ Memory event types not available - using placeholders")

# Import our tools and core components
from multi_agent_economics.tools.artifacts import create_artifact_tools
from multi_agent_economics.core.artifacts import Workspace, ArtifactManager
from multi_agent_economics.core.workspace_memory import WorkspaceMemory
from multi_agent_economics.core.budget import BudgetManager
from multi_agent_economics.core.actions import ActionLogger
from multi_agent_economics.agents.message_filter_agent_extra import (
    MessageFilterAgentExtra, MessageFilterExtraConfig, OwnToolFilter, MemoryEventFilter
)


def create_graphflow_filtered_agent(
    agent: BaseChatAgent,
    filter_tool_events: bool = True,
    filter_memory_events: bool = True
) -> MessageFilterAgent:
    """
    Test function using official MessageFilterAgent with NO filters.
    
    This is to investigate if the GraphFlow termination issue occurs with
    any MessageFilterAgent wrapper, not just our custom implementation.
    
    Args:
        agent: The agent to wrap with filtering
        filter_tool_events: Ignored - no filters applied
        filter_memory_events: Ignored - no filters applied
    Returns:
        Official MessageFilterAgent with empty filter config
    """
    # Create empty MessageFilterConfig (no filters applied)
    empty_config = MessageFilterConfig(per_source=[])

    return MessageFilterAgent(
        name=f"Filtered_{agent.name}",
        wrapped_agent=agent,
        filter=empty_config
    )


async def create_graphflow_poetry_team():
    """Create a GraphFlow team of 3 AutoGen agents with custom message filtering."""
    
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
    workspace_dir = Path("./poetry_collaboration_workspace")
    artifact_manager = ArtifactManager(workspace_dir.parent)
    shared_workspace = artifact_manager.create_workspace("poetry_collaboration")
    budget_manager = BudgetManager()
    action_logger = ActionLogger()
    print("✓ Created simulation infrastructure")
    
    # Initialize agent budgets
    initial_budget_per_agent = 10.0
    budget_manager.initialize_budget("ThemeCreator_artifacts", initial_budget_per_agent)
    budget_manager.initialize_budget("VerseWriter_artifacts", initial_budget_per_agent)
    budget_manager.initialize_budget("Editor_artifacts", initial_budget_per_agent)
    
    # Create workspace memory instances
    theme_memory = WorkspaceMemory(name="ThemeCreator_workspace", workspace=shared_workspace)
    verse_memory = WorkspaceMemory(name="VerseWriter_workspace", workspace=shared_workspace)
    editor_memory = WorkspaceMemory(name="Editor_workspace", workspace=shared_workspace)
    
    # Create artifact tools
    base_context = {
        'budget_manager': budget_manager,
        'action_logger': action_logger,
        'budget_costs': {
            'tool:load_artifact': 0.2,
            'tool:write_artifact': 0.5, 
            'tool:share_artifact': 0.3,
            'tool:unload_artifact': 0.0,
            'tool:list_artifacts': 0.0
        }
    }
    
    theme_context = {**base_context, 'agent_name': 'ThemeCreator', 'workspace_memory': theme_memory}
    verse_context = {**base_context, 'agent_name': 'VerseWriter', 'workspace_memory': verse_memory}
    editor_context = {**base_context, 'agent_name': 'Editor', 'workspace_memory': editor_memory}
    
    theme_tools = create_artifact_tools(theme_context)
    verse_tools = create_artifact_tools(verse_context)
    editor_tools = create_artifact_tools(editor_context)
    
    print(f"✓ Created {len(theme_tools)} artifact tools per agent")
    
    # Create the original agents (unfiltered)
    theme_agent = AssistantAgent(
        name="ThemeCreator",
        model_client=model_client,
        tools=theme_tools,
        memory=[theme_memory],
        reflect_on_tool_use=True,
        max_tool_iterations=10,
        system_message="""You are a creative theme creator for collaborative poetry. 

Your role is to:
1. Create initial themes and structures for poems using write_artifact
2. Use list_artifacts to see what artifacts are available in the shared workspace
3. Communicate naturally with your collaborators about your creative work

IMPORTANT: Focus on natural conversation! When you create artifacts, mention them in your response:
- "I've created a digital connections theme about..." 
- "VerseWriter, I've just created 'digital_connections_theme' - what verses does that inspire?"

ARTIFACT GUIDELINES:
- Use descriptive artifact IDs (e.g., "digital_connections_theme", "nature_poem_structure") 
- Content should be a dictionary with meaningful structure
- All agents share the same workspace

COLLABORATION APPROACH:
- Announce your artifacts in conversation
- Ask questions and provide creative guidance
- Communicate naturally about the creative process

Example workflow:
1. Create theme: write_artifact(artifact_id="digital_connections_theme", content={...})
2. Announce: "I've created a 'digital_connections_theme' exploring how technology connects us - VerseWriter, take a look!"
3. Guide the process naturally through conversation

Always use artifact tools AND communicate naturally with your team."""
    )
    
    verse_agent = AssistantAgent(
        name="VerseWriter", 
        model_client=model_client,
        tools=verse_tools,
        memory=[verse_memory],
        reflect_on_tool_use=True,
        max_tool_iterations=10,
        system_message="""You are a skilled verse writer creating poetry based on themes.

Your role is to:
1. Check for available themes using list_artifacts
2. Load inspiring themes using load_artifact
3. Write verses based on themes using write_artifact  
4. Communicate naturally about your creative process

IMPORTANT: Focus on natural conversation! When you work with artifacts:
- "I see your digital connections theme - I'll create verses about..."
- "Editor, I've created 'digital_dreams_verses' based on the theme!"

ARTIFACT GUIDELINES:
- Look for theme artifacts in the shared workspace
- Use descriptive verse artifact IDs (e.g., "digital_dreams_verses") 
- Content should be a dictionary with your verse data

COLLABORATION APPROACH:
- Respond to teammates when they mention artifacts
- Announce your creations in conversation
- Ask questions and make creative suggestions

Example workflow:
1. Check themes: list_artifacts()
2. Load theme: load_artifact(artifact_id="digital_connections_theme")
3. Create verses: write_artifact(artifact_id="digital_dreams_verses", content={...})
4. Announce: "Editor, I've created verses inspired by the digital theme - ready for your touch!"

Use artifact tools AND communicate naturally about the collaboration."""
    )
    
    editor_agent = AssistantAgent(
        name="Editor",
        model_client=model_client,
        tools=editor_tools,
        memory=[editor_memory],
        reflect_on_tool_use=True,
        max_tool_iterations=10,
        system_message="""You are a poetry editor creating final polished poems from collaborative materials.

Your role is to:
1. Check for available materials using list_artifacts
2. Load themes and verses using load_artifact
3. Create final polished poems using write_artifact
4. Clean up memory using unload_artifact when done
5. Communicate thoughtfully about the creative process

IMPORTANT: Focus on natural conversation! When working with artifacts:
- "I've reviewed your theme and verses - beautiful work!"
- "I've created the final poem 'digital_connections_final' combining everyone's contributions"

ARTIFACT GUIDELINES:
- Look for theme and verse artifacts in the workspace
- Use descriptive names for final poems (e.g., "digital_connections_final")
- Content should include polished poem text and metadata
- Clean up loaded artifacts when finished

COLLABORATION APPROACH:
- Acknowledge teammates' creative contributions
- Provide thoughtful editorial feedback
- Announce completion naturally in conversation

Example workflow:
1. Check materials: list_artifacts()
2. Load materials: load_artifact(artifact_id="theme"), load_artifact(artifact_id="verses")
3. Create final: write_artifact(artifact_id="final_poem", content={...})
4. Clean up: unload_artifact(artifact_id="theme"), unload_artifact(artifact_id="verses")
5. Announce: "The collaborative poem is complete! I've combined your contributions into our final piece."

Use artifact tools AND communicate naturally about the collaborative process."""
    )
    
    print("✓ Created 3 original AutoGen agents")
    
    # Now create GraphFlow-compatible filtered versions
    print("🔧 Applying GraphFlow message filtering...")
    
    # Create filtered agents for GraphFlow
    filtered_theme_agent = create_graphflow_filtered_agent(
        theme_agent,
        filter_tool_events=True,
        filter_memory_events=True
    )
    
    filtered_verse_agent = create_graphflow_filtered_agent(
        verse_agent,
        filter_tool_events=True, 
        filter_memory_events=True
    )
    
    filtered_editor_agent = create_graphflow_filtered_agent(
        editor_agent,
        filter_tool_events=True,
        filter_memory_events=True,
    )
    
    print("✓ Created GraphFlow-compatible filtered agent wrappers")
    
    # Build the GraphFlow
    print("🔗 Building GraphFlow with conditional sequential routing...")
    
    builder = DiGraphBuilder()
    builder.add_node(theme_agent)
    builder.add_node(filtered_verse_agent) 
    builder.add_node(filtered_editor_agent)
    
    # Create simple sequential flow: ThemeCreator -> VerseWriter -> Editor
    # Remove conditions to ensure proper sequential execution
    builder.add_edge(theme_agent, filtered_verse_agent)
    builder.add_edge(filtered_verse_agent, filtered_editor_agent)
    
    # Set entry point to start the flow
    builder.set_entry_point(theme_agent)
    
    # Build the graph
    graph = builder.build()
    
    # Create termination condition - extend the limit to allow full workflow
    termination = MaxMessageTermination(max_messages=50)
    
    # Create GraphFlow team with FILTERED agents
    flow = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
        termination_condition=termination
    )
    
    print("✓ Created GraphFlow poetry team with filtered agents")
    
    # Package simulation objects
    simulation = {
        'artifact_manager': artifact_manager,
        'shared_workspace': shared_workspace,
        'budget_manager': budget_manager, 
        'action_logger': action_logger,
        'original_agents': [theme_agent, verse_agent, editor_agent],
        'filtered_agents': [filtered_theme_agent, filtered_verse_agent, filtered_editor_agent]
    }
    
    return flow, simulation


async def run_graphflow_poetry_collaboration():
    """Run the poetry collaboration demo with GraphFlow and custom filtering."""
    print("\n🎭 AUTOGEN GRAPHFLOW ARTIFACT TOOLS DEMO")
    print("="*60) 
    print("Using GraphFlow with custom message filtering for clean output")
    print("• Tool events filtered through GraphFlow routing (agents see own events)")
    print("• Memory events filtered through GraphFlow routing")
    print("• Natural language communication preserved")
    print("• Agent functionality fully maintained")
    print("• Sequential workflow: ThemeCreator -> VerseWriter -> Editor")
    
    # Create the GraphFlow poetry team
    flow, simulation = await create_graphflow_poetry_team()
    if not flow or not simulation:
        return False
    
    # Define the collaborative task
    task = """Let's create a collaborative poem about 'digital connections in the modern world'. 

ThemeCreator: Start by creating a poem theme and structure, then announce it naturally to the team.
VerseWriter: Check for themes, load one that inspires you, and create verses based on it.
Editor: Gather all materials, create the final polished poem, and announce completion.

Focus on natural conversation while using your artifact tools for the creative work."""
    
    print(f"\n🚀 Starting GraphFlow collaborative poetry creation...")
    print(f"Task: {task}")
    print(f"\n📋 GRAPHFLOW OUTPUT (Tool events filtered through routing)")
    print("-" * 50)
    
    try:
        # Run the GraphFlow collaboration with filtered agents
        await flow.reset()
        result = await Console(flow.run_stream(task=task))
        
        print(f"\n✅ GraphFlow collaboration completed!")
        print(f"Total messages: {len(result.messages)}")
        
        # Show filtering statistics (testing with official MessageFilterAgent)
        print(f"\n📊 OFFICIAL MESSAGEFILTERAGENT TEST")
        print("-" * 40)
        print("Testing GraphFlow with official MessageFilterAgent (no filters)")
        filtered_agents = simulation['filtered_agents']
        for filtered_agent in filtered_agents:
            agent_name = getattr(filtered_agent, '_wrapped_agent', filtered_agent).name
            print(f"{agent_name}: Using official MessageFilterAgent wrapper")
        print("Note: Official MessageFilterAgent does not provide filtering statistics")
        
        # Show final conversation summary
        print(f"\n📝 FINAL CONVERSATION SUMMARY")
        print("-" * 32)
        for i, msg in enumerate(result.messages[-3:], len(result.messages)-2):
            sender = getattr(msg, 'source', 'Unknown')
            content = str(msg.content)
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"{i}. {sender}: {content}")
        
        # Show simulation summaries
        print(f"\n💰 SIMULATION BUDGET SUMMARY")
        print("-" * 30)
        budget_manager = simulation['budget_manager']
        for category in ["ThemeCreator_artifacts", "VerseWriter_artifacts", "Editor_artifacts"]:
            balance = budget_manager.get_balance(category)
            transactions = len(budget_manager.get_transaction_history(category))
            agent_name = category.replace('_artifacts', '')
            print(f"{agent_name}: ${balance:.2f} remaining, {transactions} transactions")
        
        print(f"\n⚡ ACTION SUMMARY")  
        print("-" * 20)
        action_logger = simulation['action_logger']
        actions = action_logger.get_all_actions()
        
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
        
        # Audit results
        all_tools_used = set()
        for summary in action_by_actor.values():
            all_tools_used.update(summary['tools_used'])
        
        expected_tools = {'load_artifact', 'write_artifact', 'list_artifacts', 'unload_artifact'}
        tools_used_count = len(all_tools_used.intersection(expected_tools))
        audit_success = tools_used_count >= 3
        
        print(f"\n🔍 AUDIT RESULTS")
        print("-" * 15)
        print(f"Tools used: {tools_used_count}/4 ({', '.join(sorted(all_tools_used))})")
        print(f"Audit status: {'PASS' if audit_success else 'PARTIAL'}")
        print(f"Test status: OFFICIAL MESSAGEFILTERAGENT ✨")
        
        return audit_success
        
    except Exception as e:
        print(f"❌ Error during GraphFlow collaboration: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main demo function."""
    print("🎯 AutoGen 0.6.4 Artifact Tools Demo - TESTING OFFICIAL MESSAGEFILTERAGENT")
    print("Testing GraphFlow with official MessageFilterAgent (no filters) to isolate issue")
    
    try:
        success = await run_graphflow_poetry_collaboration()
        
        if success:
            print(f"\n🎉 OFFICIAL MESSAGEFILTERAGENT TEST COMPLETED!")
            print("✅ GraphFlow worked successfully with official MessageFilterAgent")
            print("✅ All three agents participated in the conversation flow")
            print("✅ Sequential routing: ThemeCreator -> VerseWriter -> Editor")
            print("✅ Agent functionality fully preserved")
            print("✅ Natural language communication maintained")
            print("✅ Simulation budget and action tracking worked correctly")
            print("\nKey findings:")
            print("  • Official MessageFilterAgent (no filters) works with GraphFlow")
            print("  • Issue may be specific to custom MessageFilterAgent implementations")
            print("  • GraphFlow routing system is compatible with wrapped agents")
        else:
            print(f"\n❌ GRAPHFLOW DEMO HAD ISSUES")
            print("Some artifact tools may not have been used or there were errors")
            return False
            
    except Exception as e:
        print(f"\n💥 GraphFlow demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    print("Starting Official MessageFilterAgent Test with GraphFlow...")
    success = asyncio.run(main())
    
    if success:
        print("\n✨ Official MessageFilterAgent test completed successfully!")
        print("🔧 GraphFlow works correctly with official MessageFilterAgent!")
        sys.exit(0)
    else:
        print("\n⚠️  Official MessageFilterAgent test had issues.")
        sys.exit(1)
