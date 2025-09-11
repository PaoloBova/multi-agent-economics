"""
Modular tool call tracker for multi-agent experiments.
Plugs into existing experiments without disrupting core functionality.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Sequence, Union
from collections import defaultdict, Counter
from datetime import datetime

try:
    from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
    from autogen_core.models import FunctionExecutionResult
except ImportError:
    # Fallback if imports fail
    BaseAgentEvent = object
    BaseChatMessage = object
    FunctionExecutionResult = object


@dataclass
class ToolCallStats:
    """Simple statistics for tool calls."""
    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    agents_used: set = field(default_factory=set)
    
    @property
    def success_rate(self) -> float:
        return self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0


class ToolCallTracker:
    """Lightweight tracker for tool calls in multi-agent experiments."""
    
    def __init__(self):
        self.tool_stats: Dict[str, ToolCallStats] = defaultdict(lambda: ToolCallStats(""))
        self.chat_tool_calls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.period_stats: List[Dict[str, Any]] = []
    
    def process_messages(self, chat_id: str, messages: Sequence[Union[BaseAgentEvent, BaseChatMessage]]) -> None:
        """Process messages from a chat to extract tool call information."""
        period_tools = []
        
        for message in messages:
            if hasattr(message, 'type') and getattr(message, 'type') == 'ToolCallExecutionEvent':
                tool_calls = self._extract_tool_calls_from_event(message)
                period_tools.extend(tool_calls)
                
                # Update chat-specific tracking
                self.chat_tool_calls[chat_id].extend(tool_calls)
        
        # Update global stats
        for tool_call in period_tools:
            tool_name = tool_call['tool_name']
            agent_name = tool_call['agent_name']
            success = tool_call['success']
            
            if tool_name not in self.tool_stats:
                self.tool_stats[tool_name] = ToolCallStats(tool_name)
            
            stats = self.tool_stats[tool_name]
            stats.total_calls += 1
            stats.agents_used.add(agent_name)
            
            if success:
                stats.successful_calls += 1
            else:
                stats.failed_calls += 1
    
    def _extract_tool_calls_from_event(self, message) -> List[Dict[str, Any]]:
        """Extract tool call information from a ToolCallExecutionEvent."""
        tool_calls = []
        
        if not hasattr(message, 'content'):
            return tool_calls
        
        content_list = getattr(message, 'content', [])
        source = getattr(message, 'source', 'unknown')
        
        for function_result in content_list:
            if hasattr(function_result, 'name'):
                tool_call = {
                    'tool_name': function_result.name,
                    'agent_name': source,
                    'success': not getattr(function_result, 'is_error', True),
                    'timestamp': datetime.now().isoformat(),
                    'parameters': getattr(function_result, 'parameters', {}),
                    'result_preview': str(function_result.content)[:100] + "..." if len(str(function_result.content)) > 100 else str(function_result.content)
                }
                tool_calls.append(tool_call)
        
        return tool_calls
    
    def record_period_stats(self, period: int) -> None:
        """Record tool call stats for a completed period."""
        period_summary = {
            'period': period,
            'tool_calls_by_type': {name: stats.total_calls for name, stats in self.tool_stats.items()},
            'success_rates': {name: stats.success_rate for name, stats in self.tool_stats.items()},
            'total_calls': sum(stats.total_calls for stats in self.tool_stats.values()),
            'total_successes': sum(stats.successful_calls for stats in self.tool_stats.values()),
            'timestamp': datetime.now().isoformat()
        }
        self.period_stats.append(period_summary)
    
    def print_summary(self) -> None:
        """Print a concise summary of tool usage."""
        if not self.tool_stats:
            print("No tool calls tracked.")
            return
        
        print(f"\n{'='*50}")
        print("TOOL CALL SUMMARY")
        print(f"{'='*50}")
        
        # Overall stats
        total_calls = sum(stats.total_calls for stats in self.tool_stats.values())
        total_successes = sum(stats.successful_calls for stats in self.tool_stats.values())
        overall_success_rate = total_successes / total_calls * 100 if total_calls > 0 else 0
        
        print(f"Total tool calls: {total_calls}")
        print(f"Overall success rate: {overall_success_rate:.1f}%")
        print()
        
        # Per-tool breakdown
        print(f"{'Tool Name':<25} {'Calls':<8} {'Success':<8} {'Rate':<8} {'Agents':<6}")
        print("-" * 55)
        
        for tool_name, stats in sorted(self.tool_stats.items()):
            print(f"{tool_name:<25} {stats.total_calls:<8} {stats.successful_calls:<8} {stats.success_rate*100:5.1f}% {len(stats.agents_used):<6}")
        
        # Chat breakdown
        if self.chat_tool_calls:
            print(f"\nTool calls by chat:")
            for chat_id, calls in self.chat_tool_calls.items():
                success_count = sum(1 for call in calls if call['success'])
                print(f"  {chat_id}: {len(calls)} calls ({success_count} successful)")
        
        print(f"{'='*50}\n")
    
    def get_failed_calls(self) -> List[Dict[str, Any]]:
        """Return details of failed tool calls for debugging."""
        failed_calls = []
        for chat_calls in self.chat_tool_calls.values():
            for call in chat_calls:
                if not call['success']:
                    failed_calls.append(call)
        return failed_calls
    
    def print_failed_calls(self) -> None:
        """Print details of failed tool calls for debugging."""
        failed_calls = self.get_failed_calls()
        
        if not failed_calls:
            print("No failed tool calls to report.")
            return
        
        print(f"\n{'='*50}")
        print("FAILED TOOL CALLS")
        print(f"{'='*50}")
        
        for i, call in enumerate(failed_calls, 1):
            print(f"{i}. {call['tool_name']} by {call['agent_name']}")
            print(f"   Result: {call['result_preview']}")
            if call['parameters']:
                print(f"   Parameters: {call['parameters']}")
            print()