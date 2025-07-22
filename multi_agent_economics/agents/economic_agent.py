"""
Enhanced AutoGen agents with tool integration and workspace access.
"""

import json
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from ..core.artifacts import ArtifactManager, Workspace
from ..core.tools import ToolRegistry, call_tool
from ..core.actions import ActionLogger, InternalAction, ExternalAction
from ..core.budget import BudgetManager
from ..core.quality import QualityTracker


class EconomicAgent(AssistantAgent):
    """Enhanced AutoGen agent with economic simulation capabilities."""
    
    def __init__(self, name: str, model_client, role: str, organization: str,
                 artifact_manager: ArtifactManager, tool_registry: ToolRegistry,
                 budget_manager: BudgetManager, action_logger: ActionLogger,
                 quality_tracker: QualityTracker, 
                 prompt_templates_path: Optional[Path] = None,
                 role_definitions_path: Optional[Path] = None,
                 **kwargs):
        
        self.role = role
        self.organization = organization
        self.agent_id = f"{organization}.{role}"
        
        # Core managers
        self.artifact_manager = artifact_manager
        self.tool_registry = tool_registry
        self.budget_manager = budget_manager
        self.action_logger = action_logger
        self.quality_tracker = quality_tracker
        
        # Load prompt templates and role definitions
        self.prompt_templates_path = prompt_templates_path
        self.role_definitions_path = role_definitions_path
        
        # Get or create workspace
        self.workspace = artifact_manager.get_workspace(organization)
        if not self.workspace:
            self.workspace = artifact_manager.create_workspace(organization)
        
        # Build system message based on role
        system_message = self._build_system_message()
        
        # Register our custom tools
        tools = self._build_tool_functions()
        
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message,
            tools=tools,
            **kwargs
        )
    
    def _build_system_message(self) -> str:
        """Build role-specific system message from external templates."""
        
        # Load base prompt template
        base_message = self._load_base_prompt_template()
        
        # Load role-specific guidance
        role_guidance = self._load_role_guidance()
        
        return base_message + "\n\n" + role_guidance
    
    def _load_base_prompt_template(self) -> str:
        """Load base prompt template from file or use fallback."""
        if self.prompt_templates_path and (self.prompt_templates_path / "base_agent_prompt.md").exists():
            with open(self.prompt_templates_path / "base_agent_prompt.md", 'r') as f:
                template = f.read()
                return template.format(role=self.role, organization=self.organization)
        
        # Fallback template
        return f"""You are {self.role} at {self.organization}, an economic agent in a financial simulation.

Your responsibilities:
- Make strategic decisions using available tools
- Manage your budget efficiently (you have limited credits)
- Collaborate with your team by sharing artifacts
- Consider quality vs cost trade-offs in your work

Available workspace buckets:
- private: Your personal artifacts
- shared: Artifacts shared with you
- org: Organization-wide artifacts

Remember:
- Every tool call costs credits from your budget
- Higher precision tools cost more but provide better quality
- You can share artifacts with teammates (costs time but not credits)
- External actions affect the market and are visible to competitors

Your goal is to maximize your organization's profits while maintaining high quality standards."""
    
    def _load_role_guidance(self) -> str:
        """Load role-specific guidance from file or use fallback."""
        if self.role_definitions_path and self.role_definitions_path.exists():
            try:
                with open(self.role_definitions_path, 'r') as f:
                    role_definitions = json.load(f)
                
                role_info = role_definitions.get(self.role, {})
                if role_info:
                    responsibilities = "\n".join(f"- {resp}" for resp in role_info.get("responsibilities", []))
                    return f"As a {self.role}, you:\n{responsibilities}"
            except Exception:
                pass  # Fall back to hardcoded definitions
        
        # Fallback role guidance
        role_guidance = {
            "Analyst": """
As an Analyst, you:
- Research market conditions and sector forecasts
- Provide data-driven insights to your team
- Use tools like sector_forecast to gather market intelligence
- Share your analysis with Structurers and PMs""",
            
            "Structurer": """
As a Structurer, you:
- Design and price financial products
- Use forecasts from Analysts to create structured notes
- Balance quality (better pricing) with cost constraints
- Create documentation for products""",
            
            "PM": """
As a Product Manager, you:
- Coordinate team activities and manage workflows
- Make final pricing decisions and market commitments
- Monitor team progress and budget allocation
- Execute external market actions""",
            
            "Risk-Officer": """
As a Risk Officer, you:
- Assess portfolio risks and calculate VaR
- Evaluate proposed investments and products
- Provide risk assessments to support trading decisions
- Monitor market exposures""",
            
            "Trader": """
As a Trader, you:
- Execute buy/sell decisions based on risk analysis
- Monitor market prices and opportunities
- Make accept/reject decisions on offered products
- Coordinate with Risk Officers on position sizing"""
        }
        
        return role_guidance.get(self.role, "")
    
    def _build_tool_functions(self) -> List[Callable]:
        """Build AutoGen-compatible tool functions."""
        tools = []
        
        # Get all available tools from registry
        for tool_id in self.tool_registry.tools:
            tool_func = self._create_tool_wrapper(tool_id)
            tools.append(tool_func)
        
        # Add workspace management functions
        tools.extend([
            self._read_artifact,
            self._list_artifacts,
            self._share_artifact_tool,
            self._post_external_action
        ])
        
        return tools
    
    def _create_tool_wrapper(self, tool_id: str) -> Callable:
        """Create AutoGen-compatible wrapper for a tool."""
        tool = self.tool_registry.get_tool(tool_id)
        
        def tool_wrapper(**kwargs) -> str:
            """Execute tool and handle credit management."""
            # Check budget
            if not self.budget_manager.debit(self.agent_id, tool.cost, f"tool:{tool_id}"):
                return f"ERROR: Insufficient credits. Tool {tool_id} costs {tool.cost} credits."
            
            try:
                # Execute tool
                artifact = call_tool(tool_id, self.tool_registry, self.agent_id, **kwargs)
                
                # Store artifact in workspace
                self.workspace.store_artifact(artifact, bucket="private")
                
                # Log internal action
                action = InternalAction(
                    actor=self.agent_id,
                    action="call_tool",
                    tool=tool_id,
                    inputs=kwargs,
                    outputs={"artifact_id": artifact.id},
                    cost=tool.cost,
                    latency=tool.latency
                )
                self.action_logger.log_internal_action(action)
                
                # Return summary for the agent
                return f"Tool {tool_id} executed successfully. Created artifact {artifact.id} with result: {json.dumps(artifact.payload, indent=2)}"
                
            except Exception as e:
                # Refund credits on error
                self.budget_manager.credit(self.agent_id, tool.cost, f"refund:tool:{tool_id}")
                return f"ERROR executing {tool_id}: {str(e)}"
        
        # Set function metadata for AutoGen
        tool_wrapper.__name__ = tool_id
        tool_wrapper.__doc__ = f"{tool.description}\n\nCost: {tool.cost} credits\nLatency: {tool.latency} ticks"
        
        return tool_wrapper
    
    def _read_artifact(self, artifact_id: str) -> str:
        """Read an artifact from the workspace."""
        artifact = self.workspace.get_artifact(artifact_id)
        if not artifact:
            return f"Artifact {artifact_id} not found or not accessible."
        
        # Check access permissions
        if not self.artifact_manager.check_access(self.agent_id, artifact_id):
            return f"Access denied to artifact {artifact_id}."
        
        return f"Artifact {artifact_id}:\nType: {artifact.type}\nCreated by: {artifact.created_by}\nPayload: {json.dumps(artifact.payload, indent=2)}"
    
    def _list_artifacts(self, bucket: str = "all") -> str:
        """List available artifacts in workspace."""
        artifacts = self.workspace.list_artifacts(bucket)
        if not artifacts:
            return f"No artifacts found in {bucket} bucket(s)."
        
        # Get details for each artifact
        artifact_details = []
        for artifact_id in artifacts:
            artifact = self.workspace.get_artifact(artifact_id)
            if artifact and self.artifact_manager.check_access(self.agent_id, artifact_id):
                artifact_details.append({
                    "id": artifact_id,
                    "type": artifact.type,
                    "created_by": artifact.created_by,
                    "created_at": artifact.created_at.isoformat()
                })
        
        return f"Available artifacts:\n{json.dumps(artifact_details, indent=2)}"
    
    def _share_artifact_tool(self, artifact_id: str, target_agent: str) -> str:
        """Share an artifact with another agent."""
        # Extract target organization
        target_org = target_agent.split('.')[0]
        
        # Cost for sharing (time but not credits)
        success = self.artifact_manager.share_artifact(artifact_id, self.organization, target_org)
        
        if success:
            # Log internal action
            action = InternalAction(
                actor=self.agent_id,
                action="share_artifact",
                inputs={"artifact_id": artifact_id, "target": target_agent},
                cost=0.2,  # Time cost
                latency=1
            )
            self.action_logger.log_internal_action(action)
            
            return f"Successfully shared artifact {artifact_id} with {target_agent}."
        else:
            return f"Failed to share artifact {artifact_id}. Check if artifact exists and target is valid."
    
    def _post_external_action(self, action_type: str, **kwargs) -> str:
        """Post an external action that affects the market."""
        # Log external action
        action = ExternalAction(
            actor=self.agent_id,
            action=action_type,
            **kwargs
        )
        self.action_logger.log_external_action(action)
        
        return f"Posted external action: {action_type} with parameters {kwargs}"


def create_agent(name: str, role: str, organization: str, model_client,
                artifact_manager: ArtifactManager, tool_registry: ToolRegistry,
                budget_manager: BudgetManager, action_logger: ActionLogger,
                quality_tracker: QualityTracker,
                prompt_templates_path: Optional[Path] = None,
                role_definitions_path: Optional[Path] = None) -> EconomicAgent:
    """Factory function to create economic agents."""
    
    return EconomicAgent(
        name=name,
        model_client=model_client,
        role=role,
        organization=organization,
        artifact_manager=artifact_manager,
        tool_registry=tool_registry,
        budget_manager=budget_manager,
        action_logger=action_logger,
        quality_tracker=quality_tracker,
        prompt_templates_path=prompt_templates_path,
        role_definitions_path=role_definitions_path
    )
