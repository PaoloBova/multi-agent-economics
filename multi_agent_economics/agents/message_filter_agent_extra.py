from typing import AsyncGenerator, List, Sequence, Union

from autogen_core import CancellationToken, Component, ComponentModel
from pydantic import BaseModel

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage

import logging
from autogen_agentchat.messages import TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent
from autogen_agentchat.messages import MemoryQueryEvent

logger = logging.getLogger(__name__)


class MessageFilter(BaseModel):
    """Base class for message filters."""
    pass


class OwnToolFilter(MessageFilter):
    """Filter that allows agent's own tool events through but blocks others."""
    filter_requests: bool = True
    filter_executions: bool = True


class MemoryEventFilter(MessageFilter):
    """Filter that blocks memory query events from other agents."""
    allow_own: bool = True


class MessageFilterExtraConfig(BaseModel):
    """Configuration for enhanced message filtering using filter rules."""
    filters: List[MessageFilter] = []


class MessageFilterAgentExtraConfig(BaseModel):
    name: str
    wrapped_agent: ComponentModel
    filter: MessageFilterExtraConfig


class MessageFilterAgentExtra(BaseChatAgent, Component[MessageFilterAgentExtraConfig]):
    """
    A wrapper agent that filters incoming messages before passing them to the inner agent.

    This extends MessageFilterAgent functionality to support type-based filtering.
    """

    component_config_schema = MessageFilterAgentExtraConfig
    component_provider_override = "autogen_agentchat.agents.MessageFilterAgent"

    def __init__(
        self,
        name: str,
        wrapped_agent: BaseChatAgent,
        filter: MessageFilterExtraConfig,
    ):
        super().__init__(name=name, description=f"{wrapped_agent.description} (with message filtering)")
        self._wrapped_agent = wrapped_agent
        self._filter = filter
        self._original_agent_name = wrapped_agent.name
        
        # Track message statistics for debugging
        self._message_stats = {
            'total_received': 0,
            'filtered_out': 0,
            'allowed_through': 0,
            'tool_requests_filtered': 0,
            'tool_executions_filtered': 0,
            'memory_queries_filtered': 0,
            'own_tool_events_allowed': 0
        }

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return self._wrapped_agent.produced_message_types
    
    def _should_filter_by_own_tool_filter(self, filter_rule: OwnToolFilter, message: BaseChatMessage) -> bool:
        """Apply OwnToolFilter logic to a message."""
        # Always allow TextMessage - this is core communication
        if isinstance(message, TextMessage):
            return False
        
        message_source = getattr(message, 'source', '')
        is_own_message = message_source.startswith(self._original_agent_name)
        
        # Handle tool request events
        if isinstance(message, ToolCallRequestEvent):
            if not filter_rule.filter_requests:
                return False
                
            # Allow own tool events through
            if is_own_message:
                self._message_stats['own_tool_events_allowed'] += 1
                logger.debug(f"Allowing own ToolCallRequestEvent for {self._original_agent_name} from {message_source}")
                return False
            
            # Filter out other agents' tool requests
            self._message_stats['tool_requests_filtered'] += 1
            logger.debug(f"Filtering ToolCallRequestEvent from {message_source} for {self._original_agent_name}")
            return True
        
        # Handle tool execution events  
        if isinstance(message, ToolCallExecutionEvent):
            if not filter_rule.filter_executions:
                return False
                
            # Allow own tool events through
            if is_own_message:
                self._message_stats['own_tool_events_allowed'] += 1
                logger.debug(f"Allowing own ToolCallExecutionEvent for {self._original_agent_name} from {message_source}")
                return False
            
            # Filter out other agents' tool executions
            self._message_stats['tool_executions_filtered'] += 1 
            logger.debug(f"Filtering ToolCallExecutionEvent from {message_source} for {self._original_agent_name}")
            return True
        
        # Default: allow through
        return False
    
    def _should_filter_by_memory_filter(self, filter_rule: MemoryEventFilter, message: BaseChatMessage) -> bool:
        """Apply MemoryEventFilter logic to a message."""
        if isinstance(message, MemoryQueryEvent):
            message_source = getattr(message, 'source', '')
            is_own_message = message_source.startswith(self._original_agent_name)
            
            # Allow own memory events if configured
            if filter_rule.allow_own and is_own_message:
                logger.debug(f"Allowing own MemoryQueryEvent for {self._original_agent_name} from {message_source}")
                return False
                
            # Filter out memory events
            self._message_stats['memory_queries_filtered'] += 1
            logger.debug(f"Filtering MemoryQueryEvent from {message_source} for {self._original_agent_name}")
            return True
            
        # Default: allow through
        return False

    def _should_filter_message(self, message: BaseChatMessage) -> bool:
        """
        Determine if a message should be filtered out based on filter rules.

        Args:
            message: The message to evaluate
            
        Returns:
            True if message should be filtered out, False if it should be allowed through
        """
        self._message_stats['total_received'] += 1
        
        # Apply each filter rule
        for filter_rule in self._filter.filters:
            if isinstance(filter_rule, OwnToolFilter):
                if self._should_filter_by_own_tool_filter(filter_rule, message):
                    return True
            elif isinstance(filter_rule, MemoryEventFilter):
                if self._should_filter_by_memory_filter(filter_rule, message):
                    return True
            # Future filter types can be added here
            
        # Default: allow message through
        return False
    
    def _apply_filter(self, messages: Sequence[BaseChatMessage]) -> List[BaseChatMessage]:
        """Apply filtering rules to a sequence of messages."""
        filtered_messages = []
        
        for message in messages:
            if not self._should_filter_message(message):
                filtered_messages.append(message)
                self._message_stats['allowed_through'] += 1
            else:
                self._message_stats['filtered_out'] += 1
        
        # Always log filtering summary
        if len(messages) > 0:
            original_count = len(messages)
            filtered_count = len(filtered_messages)
            logger.info(f"Enhanced filtering for {self.name}: {original_count} -> {filtered_count} messages")
        
        return filtered_messages

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        filtered = self._apply_filter(messages)
        return await self._wrapped_agent.on_messages(filtered, cancellation_token)

    async def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[Union[BaseAgentEvent, BaseChatMessage, Response], None]:
        filtered = self._apply_filter(messages)
        async for item in self._wrapped_agent.on_messages_stream(filtered, cancellation_token):
            yield item

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset both the filter and the wrapped agent."""
        # Reset message statistics
        for key in self._message_stats:
            self._message_stats[key] = 0
            
        # Reset the wrapped agent
        await self._wrapped_agent.on_reset(cancellation_token)
        
        logger.info(f"Reset MessageFilterAgentExtra for {self.name}")
    
    def get_filter_stats(self) -> dict[str, int]:
        """Get message filtering statistics for debugging."""
        return self._message_stats.copy()

    def _to_config(self) -> MessageFilterAgentExtraConfig:
        return MessageFilterAgentExtraConfig(
            name=self.name,
            wrapped_agent=self._wrapped_agent.dump_component(),
            filter=self._filter,
        )

    @classmethod
    def _from_config(cls, config: MessageFilterAgentExtraConfig) -> "MessageFilterAgentExtra":
        wrapped = BaseChatAgent.load_component(config.wrapped_agent)
        return cls(
            name=config.name,
            wrapped_agent=wrapped,
            filter=config.filter,
        )