"""
Demonstration of marketing-focused multi-agent economics simulation.

This demo shows how the transformed system works where agents receive
pre-assigned forecasts and focus entirely on marketing decisions.
"""

import numpy as np
from pathlib import Path
import asyncio
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime



        
from autogen_agentchat.agents import AssistantAgent
import autogen_agentchat.conditions as conditions
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import multi_agent_economics.core.abm as abm
import multi_agent_economics.models.market_for_finance as market_for_finance
from multi_agent_economics.models.market_for_finance import (
    MarketModel, MarketState, BuyerState, SellerState, RegimeParameters
)
import multi_agent_economics.models.scenario_templates as scenario_templates
from multi_agent_economics.tools.schemas import PostToMarketResponse
from multi_agent_economics.core.artifacts import ArtifactManager
from multi_agent_economics.core.workspace_memory import WorkspaceMemory
from multi_agent_economics.tools.artifacts import create_artifact_tools
from multi_agent_economics.tools.economic import create_economic_tools
from multi_agent_economics.tools.implementations.economic import sector_forecast_impl


from typing import Sequence
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_core.models import FunctionExecutionResult

from tool_call_tracker import ToolCallTracker

def setup_logging(log_file: Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,  # Changed from DEBUG to INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Overwrite log file each run
            # logging.StreamHandler()  # Also log to console
        ]
    )

    # Reduce log volume from noisy components (filter out DEBUG/INFO, keep WARNING+)
    logging.getLogger('autogen_core.events').setLevel(logging.WARNING)  # Reduces 335+ message passing logs
    logging.getLogger('httpcore').setLevel(logging.WARNING)             # Reduces HTTP connection details
    logging.getLogger('openai._base_client').setLevel(logging.WARNING)  # Reduces OpenAI request/response details
    logging.getLogger('httpx').setLevel(logging.WARNING)                # Reduces HTTP request logs
    logging.getLogger('asyncio').setLevel(logging.WARNING)              # Reduces async task logs

    # Keep useful logs at INFO level (artifact operations, demo progress, errors)
    logger = logging.getLogger('demo_marketing_agents')
    logging.getLogger('artifacts.tools').setLevel(logging.INFO)         # Artifact tool operations
    logging.getLogger('autogen_core').setLevel(logging.INFO)            # Core AutoGen operations (non-events)
    return logger

# logger will be set up in main function
# Global tracker - will be initialized in main function
tool_tracker = None

def create_terminate_function(chat_id: str):
    """Create a terminate function for a specific chat that also tracks tool calls."""
    def terminate_chat(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> bool:
        """Determine if the chat should terminate based on the message."""
        # Track tool calls if tracker is available
        if tool_tracker:
            tool_tracker.process_messages(chat_id, messages)
        
        try:
            # Look through recent messages for successful post_to_market calls
            # Check last few messages since agents often send follow-up TextMessages after tool calls
            recent_messages = messages[-5:] if len(messages) >= 5 else messages
            
            for message in reversed(recent_messages):  # Check most recent first
                # Handle ToolCallExecutionEvent messages (the actual message type in logs)
                if hasattr(message, 'content') and hasattr(message, 'type'):
                    if getattr(message, 'type', None) == 'ToolCallExecutionEvent':
                        # ToolCallExecutionEvent.content is a list of FunctionExecutionResult objects
                        content_list = getattr(message, 'content', [])
                        for function_result in content_list:
                            if (hasattr(function_result, 'name') and 
                                hasattr(function_result, 'content') and
                                function_result.name == "post_to_market" and 
                                not getattr(function_result, 'is_error', True)):
                                
                                # Parse the JSON content to check status
                                try:
                                    result_data = PostToMarketResponse.model_validate_json(function_result.content)
                                    if result_data.status == "success":
                                        return True
                                except:
                                    # If content is already parsed object, try direct access
                                    import json
                                    try:
                                        if isinstance(function_result.content, str):
                                            content_dict = json.loads(function_result.content)
                                        else:
                                            content_dict = function_result.content
                                        if content_dict.get("status") == "success":
                                            return True
                                    except:
                                        continue
                
                # Legacy handling for direct FunctionExecutionResult (just in case)
                elif hasattr(message, 'name') and message.name == "post_to_market":
                    try:
                        content = PostToMarketResponse.model_validate_json(message.content)
                        if content.status == "success":
                            return True
                    except:
                        continue
            
            return False
        except Exception as e:
            # More informative error handling for debugging
            print(f"terminate_chat error: {e}")
            return False
    
    return terminate_chat

def create_market_state():
    """Create a market state for demonstration."""

    # Create buyers with diverse preferences
    buyers = []
    for i in range(10):
        # Vary buyer preferences and budgets - scale weights to match budget range
        methodology_weight = (0.4 + (i % 3) * 0.2) * 100  # 40, 60, 80
        coverage_weight = (0.3 + (i % 2) * 0.4) * 100     # 30, 70  
        budget = 80 + i * 10                              # 80-170
        
        buyer = BuyerState(
            buyer_id=f"buyer_{i}",
            regime_beliefs={"tech": [0.6, 0.4]},
            budget=budget,
            attr_weights={"tech": [methodology_weight, coverage_weight]},
            attr_mu={"tech": [methodology_weight, coverage_weight]},
            attr_sigma2={"tech": [0.2, 0.2]},
            buyer_conversion_function={
                "methodology": {"premium": 0.9, "standard": 0.6, "basic": 0.3},
                "coverage": {"numeric_scaling": True, "base": 0.1, "scale": 0.8}
            }
        )
        buyers.append(buyer)
    
    # Create sellers with differentiated financial starting positions
    # reuters_analytics: struggling company with poor historical performance
    # forest_forecasts: dominant market leader
    sellers = []
    reuters = SellerState(
        org_id="reuters_analytics",
        production_cost=0.0,  # Will be calculated based on period costs
        surplus=0.0,  # Current period starts at 0
        total_profits=0.0  # Will be set by historical trade generation based on actual performance
    )
    sellers.append(reuters)
    
    forest = SellerState(
        org_id="forest_forecasts", 
        production_cost=0.0,  # Will be calculated based on period costs
        surplus=0.0,
        total_profits=0.0  # Will be set by historical trade generation based on actual performance
    )
    sellers.append(forest)
    
    # Regime parameters for forecasting
    regime_params = {
        "tech": {
            0: RegimeParameters(mu=0.06, sigma=0.15),  # Normal market
            1: RegimeParameters(mu=-0.02, sigma=0.25)  # Volatile market
        }
    }
    
    scenario = scenario_templates.generate_boom_scenario(sectors=["tech"], growth_factor=1.1)
    
    max_periods = 1000
    regime_history = market_for_finance.generate_regime_history_from_scenario(scenario, max_periods)
    
    # Create market state
    market_state = MarketState(
        regime_history=regime_history,
        offers=[],
        trades=[],
        index_values={"tech": 100.0},
        all_trades=[],
        buyers_state=buyers,
        sellers_state=sellers,
        current_regimes={"tech": 0},
        regime_parameters=regime_params,
        regime_correlations=np.array([[1.0]]),
        current_period=0,
        knowledge_good_forecasts={},
        knowledge_good_impacts={},
        attribute_order=["methodology", "coverage"],
        sector_order=["tech"],
        marketing_attribute_definitions={
            "methodology": {
                "type": "qualitative",
                "values": ["basic", "standard", "premium"],
                "descriptions": {
                    "premium": "Advanced machine learning methods",
                    "standard": "Traditional econometric approaches", 
                    "basic": "Simple heuristic methods"
                }
            },
            "coverage": {
                "type": "numeric",
                "range": [0.0, 1.0],
                "description": "Fraction of available data sources utilized"
            }
        },
        risk_free_rate=0.03
    )
    
    return market_state

def create_model_client_openai(args) -> OpenAIChatCompletionClient:
    """Create an OpenAI model client."""
        # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please add your OpenAI API key to .env file")
        raise ValueError("OPENAI_API_KEY not found in environment")
    # Create OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model=args["model_name"],
        api_key=api_key
    )
    return model_client

def create_model_client_anthropic(args) -> AnthropicChatCompletionClient:
    """Create an Anthropic model client."""
        # Check for Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        print("Please add your ANTHROPIC API key to .env file")
        raise ValueError("ANTHROPIC_API_KEY not found in environment")
    # Create ANTHROPIC_API_KEY model client
    model_client = AnthropicChatCompletionClient(
        model=args["model_name"],
        api_key=api_key
    )
    return model_client

def create_model_client(args):
    """Create a model client based on the specified type."""
    if args["model_type"] == "openai":
        return create_model_client_openai(args)
    elif args["model_type"] == "anthropic":
        return create_model_client_anthropic(args)
    else:
        raise ValueError(f"Unsupported model type: {args['model_type']}")

def generate_marketing_attribute_example(marketing_definitions: dict) -> dict:
    """Generate a proper marketing attributes example based on current definitions."""
    example_attrs = {}
    
    for attr_name, attr_def in marketing_definitions.items():
        attr_type = attr_def.get('type', 'qualitative')
        
        if attr_type == 'qualitative':
            # For methodology: use premium as best practice example
            if attr_name == 'methodology':
                example_attrs[attr_name] = 'premium'
            else:
                values = attr_def.get('values', ['basic', 'standard', 'premium'])
                example_attrs[attr_name] = values[-1] if values else 'premium'  # Use highest quality
        elif attr_type == 'numeric':
            # For coverage: use high-quality example 
            attr_range = attr_def.get('range', [0.0, 1.0])
            example_attrs[attr_name] = attr_range[0] + 0.8 * (attr_range[1] - attr_range[0])  # 80% of range
        else:
            example_attrs[attr_name] = 'standard'
    
    return example_attrs

def generate_effort_guidelines(config: dict) -> str:
    """Generate effort allocation guidelines based on config thresholds."""
    tool_params = config.get('tool_parameters', {})
    guidelines = []
    
    for tool_name, params in tool_params.items():
        thresholds = params.get('effort_thresholds', {})
        if thresholds:
            high_threshold = thresholds.get('high', 5.0)
            medium_threshold = thresholds.get('medium', 2.0)
            guidelines.append(f"- {tool_name}: high quality ≥{high_threshold}, medium ≥{medium_threshold}")
    
    return '\n'.join(guidelines) if guidelines else "- Use effort 2-5 for most tools (higher = better quality)"

def generate_tool_examples(market_state, config: dict) -> str:
    """Generate dynamic tool usage examples based on current market state."""
    marketing_attrs = generate_marketing_attribute_example(market_state.marketing_attribute_definitions)
    effort_guide = generate_effort_guidelines(config)
    
    examples = f"""
## Critical Tool Usage Patterns

### 1. Always Include All Required Parameters
✅ CORRECT - post_to_market with all required parameters:
```
{{"forecast_id": "forecast_abc123", "price": 750.0, "marketing_attributes": {marketing_attrs}}}
```

✅ CORRECT - research_competitive_pricing with all required parameters:
```
{{"sector": "tech", "effort": 3.0, "marketing_attributes": {marketing_attrs}}}
```

❌ WRONG - post_to_market missing marketing_attributes parameter:
```
{{"forecast_id": "forecast_abc123", "price": 750.0}}
```

❌ WRONG - research_competitive_pricing missing marketing_attributes:
```
{{"sector": "tech", "effort": 3.0}}
```

### 2. Marketing Attributes Must Match Schema
✅ CORRECT - proper marketing attributes format:
```
{marketing_attrs}
```

❌ WRONG - empty or malformed attributes:
```
{{}} or {{"invalid_key": "value"}}
```

### 3. Tool Usage Workflow
Recommended sequence for effective market research:
1. **Create forecast**: sector_forecast(sector="tech", effort=3.0)
2. **Research buyer preferences**: analyze_buyer_preferences(sector="tech", effort=3.0)  
3. **Research competitive pricing**: research_competitive_pricing(sector="tech", effort=3.0, marketing_attributes=YOUR_PRODUCT_ATTRS)
4. **Post offer**: post_to_market(forecast_id="abc123", price=750.0, marketing_attributes=YOUR_PRODUCT_ATTRS)

### 4. Effort Allocation Guidelines
{effort_guide}

### 5. Required Parameters by Tool
- post_to_market: forecast_id, price, marketing_attributes
- research_competitive_pricing: sector, effort, marketing_attributes (YOUR product attributes)
- analyze_buyer_preferences: sector, effort
- analyze_historical_performance: sector, effort
- sector_forecast: sector, effort

### 6. Common Mistake: research_competitive_pricing Usage
❌ WRONG: Using research_competitive_pricing for general market research
✅ CORRECT: Using research_competitive_pricing to price YOUR specific product

The research_competitive_pricing tool simulates buyer behavior comparing YOUR proposed 
product (with your specified marketing_attributes) against existing competitors to 
find optimal pricing. You must specify the exact attributes of the product you want to price.
"""
    
    return examples

def generate_equilibrium_historical_trades(market_state, config: dict) -> None:
    """
    Generate historical trades using theoretical equilibrium pricing.
    
    Args:
        market_state: MarketState object to populate with historical data
        config: Configuration dict containing historical parameters and market setup
    """
    import numpy as np
    import uuid
    from multi_agent_economics.models.market_for_finance import convert_marketing_to_features, TradeData
    
    num_periods = config.get("historical_periods", 50)
    org_ids = config.get("org_ids", ["forest_forecasts", "reuters_analytics"])
    quality_distribution = config.get("quality_distribution", [0.2, 0.3, 0.5])
    quality_types = ["high_quality", "medium_quality", "low_quality"]
    
    print(f"Generating {num_periods} periods of theoretical equilibrium trades...")
    
    # Define quality combinations using existing mappings from sector_forecast_impl
    quality_combinations = [
        {"methodology": "premium", "coverage_range": (0.8, 1.0)},   # high_quality
        {"methodology": "standard", "coverage_range": (0.5, 0.7)},  # medium_quality  
        {"methodology": "basic", "coverage_range": (0.1, 0.4)}      # low_quality
    ]
    
    buyers = market_state.buyers_state
    sector = "tech"
    
    # Generate historical trades for each period
    historical_trades = []
    
    for period in range(num_periods):
        # Both orgs offer identical quality+coverage this period
        quality_idx = np.random.choice(len(quality_types), p=quality_distribution)
        combo = quality_combinations[quality_idx]
        
        # Sample ONE coverage value for both orgs this period
        coverage_min, coverage_max = combo["coverage_range"]
        coverage = np.random.uniform(coverage_min, coverage_max)
        
        marketing_attributes = {
            "methodology": combo["methodology"],
            "coverage": coverage
        }
        
        # Calculate WTP for each buyer for THIS specific combination
        buyer_wtps = []
        for buyer in buyers:
            features = convert_marketing_to_features(
                marketing_attributes,
                buyer.buyer_conversion_function,
                market_state.attribute_order
            )
            wtp = np.dot(buyer.attr_weights[sector], features)
            buyer_wtps.append((wtp, buyer.budget, buyer))
        
        # Find revenue-maximizing equilibrium price for this period's combination
        buyer_wtps.sort(reverse=True)  # Sort by WTP descending
        best_revenue = 0
        best_price = 0
        
        for wtp, budget, _ in buyer_wtps:
            # Try this WTP as potential price
            price = wtp
            # Count buyers who would buy (WTP >= price AND budget >= price)
            quantity = sum(1 for w, b, _ in buyer_wtps if w >= price and b >= price)
            revenue = price * quantity
            
            if revenue > best_revenue:
                best_revenue = revenue
                best_price = price
        
        equilibrium_price = best_price
        
        # Find buyers who would purchase at equilibrium price
        eligible_buyers = []
        for wtp, budget, buyer in buyer_wtps:
            if wtp >= equilibrium_price and budget >= equilibrium_price:
                eligible_buyers.append(buyer)
        
        # Split eligible buyers with imbalanced market shares to create competitive pressure
        # forest_forecasts gets 70% market share (dominant position)
        # reuters_analytics gets 30% market share (struggling position)
        
        num_eligible = len(eligible_buyers)
        print("Eligible buyers this period:", num_eligible)
        forest_share = int(0.70 * num_eligible)  # 70% to forest_forecasts
        reuters_share = num_eligible - forest_share  # remaining 30% to reuters_analytics
        
        # Shuffle buyers to ensure random allocation within the percentage splits
        import random
        random.shuffle(eligible_buyers)
        
        org_buyer_allocations = {
            "forest_forecasts": eligible_buyers[:forest_share],
            "reuters_analytics": eligible_buyers[forest_share:forest_share + reuters_share]
        }
        
        # Create trades and track revenue for financial pressure calculation
        period_revenues = {}
        for org_id, buyer_subset in org_buyer_allocations.items():
            period_revenue = 0.0
            for buyer in buyer_subset:
                trade = TradeData(
                    buyer_id=buyer.buyer_id,
                    seller_id=org_id,
                    price=equilibrium_price,
                    quantity=1,
                    good_id=f"historical_{org_id}_{period}_{uuid.uuid4().hex[:8]}",
                    sector=sector,
                    marketing_attributes=marketing_attributes,
                    buyer_conversion_used={},  # Not needed for historical data
                    period=period
                )
                historical_trades.append(trade)
                period_revenue += equilibrium_price
            period_revenues[org_id] = period_revenue
    
    # Set simple hardcoded financial positions to create adverse selection pressure
    for seller in market_state.sellers_state:
        if seller.org_id == "reuters_analytics":
            seller.total_profits = -200.0  # Struggling company
        else:  # forest_forecasts
            seller.total_profits = 400.0   # Dominant company
    
    # Populate market state with historical data
    market_state.all_trades.extend(historical_trades)
    print(f"Generated {len(historical_trades)} trades with 70/30 market split")

def generate_simple_historical_trades(market_state, config: dict) -> None:
    """
    Generate simple, consistent historical trades representing stable duopoly equilibrium.
    
    Uses identical high-quality forecasts with 50/50 market split and healthy profit margins.
    This provides realistic data for backward-looking tools without complex market dynamics.
    """
    import numpy as np
    import uuid
    from multi_agent_economics.models.market_for_finance import convert_marketing_to_features, TradeData
    
    num_periods = config.get("historical_periods", 50)
    production_cost = config.get("historical_production_cost", 30.0)
    org_ids = config.get("org_ids", ["forest_forecasts", "reuters_analytics"])
    
    print(f"Generating {num_periods} periods of simple duopoly historical trades...")
    
    # Use consistent high-quality attributes (identical products)
    marketing_attributes = {
        "methodology": "premium", 
        "coverage": 0.85
    }
    
    buyers = market_state.buyers_state
    sector = "tech"
    
    # Calculate equilibrium price for this quality level
    buyer_wtps = []
    for buyer in buyers:
        features = convert_marketing_to_features(
            marketing_attributes,
            buyer.buyer_conversion_function,
            market_state.attribute_order
        )
        wtp = np.dot(buyer.attr_weights[sector], features)
        buyer_wtps.append((wtp, buyer.budget, buyer))
    
    # Find revenue-maximizing equilibrium price
    buyer_wtps.sort(reverse=True)
    best_revenue = 0
    best_price = 0
    
    for wtp, budget, _ in buyer_wtps:
        price = wtp
        quantity = sum(1 for w, b, _ in buyer_wtps if w >= price and b >= price)
        revenue = price * quantity
        
        if revenue > best_revenue:
            best_revenue = revenue
            best_price = price
    
    equilibrium_price = best_price
    
    # Find buyers who purchase at equilibrium price
    eligible_buyers = []
    for wtp, budget, buyer in buyer_wtps:
        if wtp >= equilibrium_price and budget >= equilibrium_price:
            eligible_buyers.append(buyer)
    
    print(f"Equilibrium price: ${equilibrium_price:.2f}")
    print(f"Production cost: ${production_cost:.2f}")  
    print(f"Profit margin: {((equilibrium_price - production_cost) / equilibrium_price * 100):.1f}%")
    print(f"Eligible buyers per period: {len(eligible_buyers)}")
    
    # Generate historical trades with 50/50 split (identical products)
    historical_trades = []
    org_revenues = {org_id: 0.0 for org_id in org_ids}
    
    for period in range(num_periods):
        # 50/50 split for identical products
        import random
        random.shuffle(eligible_buyers)
        split_point = len(eligible_buyers) // 2
        
        org_buyer_allocations = {
            org_ids[0]: eligible_buyers[:split_point],
            org_ids[1]: eligible_buyers[split_point:]
        }
        
        # Small price variation (±2%) around equilibrium
        period_price = equilibrium_price * (1 + random.uniform(-0.02, 0.02))
        
        for org_id, buyer_subset in org_buyer_allocations.items():
            for buyer in buyer_subset:
                trade = TradeData(
                    buyer_id=buyer.buyer_id,
                    seller_id=org_id,
                    price=period_price,
                    quantity=1,
                    good_id=f"hist_{org_id}_{period}_{uuid.uuid4().hex[:8]}",
                    sector=sector,
                    marketing_attributes=marketing_attributes,
                    buyer_conversion_used={},
                    period=period
                )
                historical_trades.append(trade)
                org_revenues[org_id] += period_price
    
    # Calculate total profits from actual historical performance
    for seller in market_state.sellers_state:
        org_id = seller.org_id
        if org_id in org_revenues:
            revenue = org_revenues[org_id]
            costs = production_cost * (revenue / equilibrium_price)  # costs = production_cost * quantity_sold
            seller.total_profits = revenue - costs
            print(f"{org_id}: Revenue=${revenue:.0f}, Costs=${costs:.0f}, Profit=${seller.total_profits:.0f}")
    
    # Populate market state with historical data
    market_state.all_trades.extend(historical_trades)
    print(f"Generated {len(historical_trades)} historical trades (stable duopoly equilibrium)")

def plot_historical_market_trends(all_trades: list) -> None:
    """
    Visualize historical trading patterns by quality and pricing.
    
    Args:
        all_trades: List of TradeData objects containing historical transaction data
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    if not all_trades:
        print("No historical trades to visualize")
        return
    
    print(f"Visualizing trends from {len(all_trades)} historical trades...")
    
    # Convert trade data to DataFrame for easier analysis
    trade_data = []
    for trade in all_trades:
        marketing_attrs = getattr(trade, 'marketing_attributes', {})
        methodology = marketing_attrs.get('methodology', 'unknown')
        coverage = marketing_attrs.get('coverage', 0.0)
        
        trade_data.append({
            'period': trade.period,
            'price': trade.price,
            'methodology': methodology,
            'coverage': coverage,
            'seller_id': trade.seller_id
        })
    
    df = pd.DataFrame(trade_data)
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Historical Market Analysis', fontsize=16)
    
    # 1. Trade volume by methodology over time
    ax1 = axes[0, 0]
    methodology_counts = df.groupby(['period', 'methodology']).size().unstack(fill_value=0)
    for methodology in ['basic', 'standard', 'premium']:
        if methodology in methodology_counts.columns:
            ax1.plot(methodology_counts.index, methodology_counts[methodology], 
                    label=methodology, marker='o', markersize=3)
    ax1.set_title('Trade Volume by Methodology Over Time')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Number of Trades')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price distribution by methodology
    ax2 = axes[0, 1]
    for methodology in ['basic', 'standard', 'premium']:
        subset = df[df['methodology'] == methodology]
        if not subset.empty:
            ax2.hist(subset['price'], alpha=0.6, label=methodology, bins=10)
    ax2.set_title('Price Distribution by Methodology')
    ax2.set_xlabel('Price')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Quality distribution
    ax3 = axes[1, 0]
    methodology_counts = df['methodology'].value_counts()
    colors = {'basic': 'red', 'standard': 'orange', 'premium': 'green'}
    wedge_colors = [colors.get(method, 'gray') for method in methodology_counts.index]
    ax3.pie(methodology_counts.values, labels=methodology_counts.index, autopct='%1.1f%%',
            colors=wedge_colors)
    ax3.set_title('Historical Quality Distribution')
    
    # 4. Price vs Coverage scatter
    ax4 = axes[1, 1]
    for methodology in ['basic', 'standard', 'premium']:
        subset = df[df['methodology'] == methodology]
        if not subset.empty:
            ax4.scatter(subset['coverage'], subset['price'], 
                       label=methodology, alpha=0.6, s=20)
    ax4.set_title('Price vs Coverage by Methodology')
    ax4.set_xlabel('Coverage')
    ax4.set_ylabel('Price')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Historical Market Summary ===")
    print(f"Total trades: {len(all_trades)}")
    print(f"Average price: ${df['price'].mean():.2f}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    methodology_stats = df.groupby('methodology').agg({
        'price': ['count', 'mean', 'std'],
        'coverage': ['mean']
    }).round(2)
    print("\nBreakdown by methodology:")
    print(methodology_stats)

def set_financial_pressure_budgets(model, org_ids):
    """Simple binary budget allocation based on financial performance."""
    for org_id in org_ids:
        seller = next((s for s in model.state.sellers_state if s.org_id == org_id), None)
        if seller and seller.total_profits < 0:
            model.state.budgets[org_id] = 30.0  # Struggling company
        else:
            model.state.budgets[org_id] = 60.0  # Successful company

def get_org_financial_description(org_id: str, model) -> str:
    """Simple binary org descriptions based on financial performance."""
    seller = next((s for s in model.state.sellers_state if s.org_id == org_id), None)
    if seller and seller.total_profits < 0:
        return "A struggling financial forecasting company facing potential bankruptcy. URGENT performance improvement needed."
    else:
        return "The market-leading financial forecasting company with strong profitability."

def create_agents(model, config):
    """Create agents and added context variables."""

    quality_distribution = config['quality_distribution']
    quality_types = list(config['quality_type_effort_mapping'].keys())
    org_ids = config['org_ids']
    sectors = config['sectors']
    model_clients = config['model_clients']
    n_agents = config['n_agents']
    
    agents = []
    agent_metadata = {}

    base_assistant_agent_config = {
        "reflect_on_tool_use": True,
        "max_tool_iterations": 10,
    }
    artifact_manager_path = Path(f"./demo_marketing_agents_workspaces/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    artifact_manager = ArtifactManager(artifact_manager_path)
    
    for agent_id in range(n_agents):
        model_client_data = np.random.choice(model_clients)
        model_name, model_type = model_client_data.get("model_name"), model_client_data.get("model_type")
        org_id = np.random.choice(org_ids)
        quality_type = np.random.choice(quality_types, p=quality_distribution)
        sector = np.random.choice(sectors)
        agent_name = f"seller_{agent_id}_{org_id}"
        
        model_client = create_model_client({"model_type": model_type, "model_name": model_name})
        
        tool_usage_examples = generate_tool_examples(model.state, config)
        org_description = get_org_financial_description(org_id, model)
        system_message = Path("./scripts/prompt_templates/system_prompt_marketing_task.md").read_text().format(
            org_name=org_id,
            org_description=org_description,
            marketing_attributes=model.state.marketing_attribute_definitions,
            tool_usage_examples=tool_usage_examples
        )
        reflect_on_tool_use = base_assistant_agent_config["reflect_on_tool_use"]
        max_tool_iterations = base_assistant_agent_config["max_tool_iterations"]

        agent_workspace = artifact_manager.create_workspace(f"agent_{agent_id}_{agent_name}")
        workspace_memory = WorkspaceMemory(name=f"{agent_name} personal workspace", workspace=agent_workspace)
        base_context = {
            'budget_costs': {
                'tool:load_artifact': 0.0,
                'tool:write_artifact': 0.0, 
                'tool:share_artifact': 0.0,
                'tool:unload_artifact': 0.0,
                'tool:list_artifacts': 0.0
            }
        }
        agent_context = {**base_context,
                         **config,
                         'org_id': org_id,
                         'agent_name': agent_name,
                         'workspace_memory': workspace_memory}
        tools = create_artifact_tools(agent_context)
        tools += create_economic_tools(model, agent_context)
        
        agent = AssistantAgent(name=agent_name,
                               model_client=model_client,
                               tools=tools,
                               memory=[workspace_memory],
                               reflect_on_tool_use=reflect_on_tool_use,
                               max_tool_iterations=max_tool_iterations,
                               system_message=system_message)
        agents.append(agent)
        agent_metadata[agent_name] = {
            'agent_index': agent_id,
            'agent_name': agent_name,
            'org_id': org_id,
            'quality_type': quality_type,
            'sector': sector,
            'model_name': model_name,
        }
    
    print(f"Created {len(agents)} agents")
    print(f"Quality distribution: {quality_distribution}")
    
    # Set dynamic budgets based on financial performance to create pressure
    set_financial_pressure_budgets(model, org_ids)
    
    return agents, agent_metadata

def collect_stats_demo(model, config):
    """Collect statistics for demonstration."""
    
    period = model.state.current_period
    print(f"--- Period {period + 1} ---")
    print(f"Trades executed: {len(model.state.trades)}")
    
    # Logging in this case means both to log to file and to collect in-memory
    # for summary statistics at the end of the simulation by storing in the
    # models model_results (and agent_results) attributes, appended as a dict
    # for each period (and each agent).
    
    # Log stats for the forecast features generated by each agent this period
    # Count by band of confidence level
    
    # Log stats for the offers posted by each agent this period
    # Count by band of marketing characteristics and price ranges
    
    # Log stats for the trades executed this period
    # Count by band of marketing characteristics and price ranges
    
    # Collect trade counts by quality type
    quality_types = list(config['quality_type_effort_mapping'].keys())
    trade_counts = {qtype: 0 for qtype in quality_types}
    for trade in model.state.trades:
        org_id = trade.seller_id
        # Find the first agent belonging to this org_id
        agent_id = model.chats.get(f"org_chat_{org_id}")._participants[0].name
        agent_metadata = model.agent_metadata.get(agent_id, {})
        trade_counts[agent_metadata['quality_type']] += 1
    stats = {"trade_counts": trade_counts}
    model.model_results.append(stats)

    # Record tool call stats for this period
    if tool_tracker:
        tool_tracker.record_period_stats(period)

    return

def assign_forecasts_to_agents(market_model: MarketModel, config: dict):
    """Assign forecasts to agents based on their quality type."""
    true_next_regime = market_model.state.regime_history[
        market_model.state.current_period + 1].regimes['tech']
    for _agent_id, agent_metadata in market_model.agent_metadata.items():
        # Assign forecasts based on quality type
        quality_type = agent_metadata['quality_type']
        effort = config['quality_type_effort_mapping'][quality_type]
        sector = "tech"
        sector_forecast_response = sector_forecast_impl(market_model, config, sector, 1, effort)
        agent_metadata['assigned_forecast'] = sector_forecast_response
    
    return

def create_ai_chats(market_model: MarketModel, config: dict):
    """Create AI chats to persist agent conversations."""
    
    logger = logging.getLogger('demo_marketing_agents')
    logger.info("Creating AI chats")
        
    # In this demo, agents are sellers powered by LLMs to make marketing decisions.

    # Each LLM works independently on behalf of a unique seller to market their
    # pre-assigned forecasts.
    
    # AI agents typically work in teams to evaluate and reason about how to
    # allocate their resources to different tools. In this demo, resources are
    # spent on better information and analysis about the market and buyers.
    
    # We use a simple Augoten group chat to allow the agents to build up a
    # shared understanding of the market and their strategy over time.
    
    def group_agents_by_metadata_key(model, key):
        """Group agents by a specified metadata key."""
        groups = {}
        for metadata in model.agent_metadata.values():
            agent = model.agents[metadata['agent_index']]
            group = metadata[key]
            if group not in groups:
                groups[group] = []
            groups[group].append(agent)
        return groups

    # Group agents by organization id
    # We have a chat per organization
    chats = {}

    agent_groups_by_org = group_agents_by_metadata_key(market_model, 'org_id')
    for org_id, group in agent_groups_by_org.items():
        # Create a chat for the organization
        chat_id = f"org_chat_{org_id}"
        termination_condition = conditions.FunctionalTermination(create_terminate_function(chat_id))
        chats[chat_id] = RoundRobinGroupChat(participants=group,
                                             max_turns=config["max_chat_turns"],
                                             termination_condition=termination_condition)
    
    # We also have a single-agent chat per agent that we can use if necessary.
    # Typically, we let agents terminate these chats themselves by calling a
    # stop_chat tool they have access to. If they don't have access to that tool,
    # then we give them a max turn limit.
    # TODO: Consider setting a max token limit instead of a max turn limit.
    
    for agent in market_model.agents:
        chat_id = f"single_agent_chat_{agent.name}"
        chats[chat_id] = RoundRobinGroupChat(participants=[agent],
                                   max_turns=config["max_chat_turns_single_agent"],
                                   termination_condition=conditions.StopMessageTermination())

    return chats

def derive_org_performance(market_model: MarketModel, org_id: str) -> str:
    """Simple organization performance with financial pressure."""
    seller = next((s for s in market_model.state.sellers_state if s.org_id == org_id), None)
    if not seller:
        return f"No data for {org_id}"
    
    if seller.total_profits < 0:
        return f"⚠️ {org_id}: ${abs(seller.total_profits):.0f} losses. URGENT: Company needs strong sales to survive!"
    else:
        return f"✅ {org_id}: ${seller.total_profits:.0f} profits. Maintain market leadership position."

def derive_public_market_info(market_model: MarketModel, org_id: str) -> str:
    """Simple public market information."""
    return "Market dominated by forest_forecasts (~70% share). reuters_analytics struggling with ~30% share."

async def run_ai_agents_async(market_model: MarketModel, config: dict):
    """Run the AI agents to make marketing decisions."""
    
    logger = logging.getLogger('demo_marketing_agents')
    logger.info(f"Starting run_ai_agents with {len(market_model.chats)} chats")
    
    # Start with group chats only. Run through each chat in sequence
    
    for chat_id, chat in market_model.chats.items():
        logger.info(f"Processing chat {chat_id}, type: {type(chat_id)}")
        if chat_id.startswith("org_chat_"):
            logger.info(f"Running chat {chat_id} with {len(chat._participants)} agents.")
            print(f"Running chat {chat_id} with {len(chat._participants)} agents.")
            budget_balance = 10.0  # For demo, we just refill balance each period
            org_id = market_model.agent_metadata[chat._participants[0].name]['org_id']
            org_performance = derive_org_performance(market_model, org_id)
            market_info_public = derive_public_market_info(market_model, org_id)
            current_forecast = market_model.agent_metadata[chat._participants[0].name]['assigned_forecast']
            template_file = f"marketing_task_{config.get('prompt_variant', 'subtle')}.md"
            task_prompt = Path(f"./scripts/prompt_templates/{template_file}").read_text().format(
                budget_balance=budget_balance,
                org_performance=org_performance,
                market_info_public=market_info_public,
                current_forecast=current_forecast
            )
            logger.info(f"About to run chat {chat_id} with task prompt")
            result = await Console(chat.run_stream(task=task_prompt))  # Stream the messages to the console.
        
            logger.info(f"Chat {chat_id} completed.")
            print(f"Chat {chat_id} completed.")
        else:
            logger.info(f"Skipping chat {chat_id} (not org_chat)")
    
    # We are only running org group chats which perfectly seperate agents. So,
    # we have no need to save and load state between chats and between rounds.
    
    # TODO: Validate that all groups have posted an offer. If not, raise a
    # warning.
    
    logger.info("run_ai_agents completed")
    return

async def run_ai_agents(market_model: MarketModel, config: dict):
    """Run the AI agents to make marketing decisions."""
    await run_ai_agents_async(market_model, config)
    return

async def run_model_step(market_model: MarketModel, config: dict):
    """Run a single step of the market model."""
    
    logger = logging.getLogger('demo_marketing_agents')
    logger.info(f"run_model_step called for tick {market_model.tick}")
    assign_forecasts_to_agents(market_model, config)
    logger.info("Forecasts assigned, about to run AI agents")

    await run_ai_agents(market_model, config)

    # Each LLM seller should have posted a trade offer by now. If they haven't,
    # raise a warning.
    print("Number of offers posted:", len(market_model.state.offers))
    for agent in market_model.agents:
        org_id = market_model.agent_metadata[agent.name]['org_id']
        if not any(offer.seller_id == org_id
                   for offer in market_model.state.offers):
            print(f"WARNING: Seller {org_id} did not post any offers this period.")
    
    # Run the market simulation step
    print("Running market simulation step...")
    market_for_finance.model_step(market_model, config)
    print("Market simulation step completed.")
    
    return

async def run_demo_simulation_async():
    """Run a demonstration of the marketing-focused simulation."""
    
    print("=== Marketing-Focused Multi-Agent Economics Demo ===\n")
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging for debugging
    log_file = Path("./demo_marketing_agents.log")
    print(f"Setting up logging to: {log_file.absolute()}")
    logger = setup_logging(log_file)
    print("Logging setup complete")
    logger.info("Demo simulation starting")
    print("Logged demo starting message")
    
    # Initialize tool call tracker
    global tool_tracker
    tool_tracker = ToolCallTracker()
    print("Tool call tracker initialized")
    
    # Initialise model

    config = {
        'sector': 'tech',
        'horizon': 1,
        'effort_distributions': {
            'high_quality': {'mean': 7.0, 'std': 1.0},
            'medium_quality': {'mean': 4.0, 'std': 1.0},
            'low_quality': {'mean': 1.5, 'std': 0.5}
        },
        'quality_type_effort_mapping': {
            'high_quality': 5.0,
            'medium_quality': 3.0,
            'low_quality': 1.0
        },
        
        "quality_distribution": [0.2, 0.3, 0.5],  # Probabilities for high, medium, low quality
        "org_ids": ['forest_forecasts', 'reuters_analytics',
                    'alpha_insights', 'beta_data_solutions',
                    'gamma_market_research', 'delta_financials',
                    'epsilon_analytics', 'zeta_forecasting',
                    'theta_data_services', 'iota_market_intel'],
        "sectors": ['tech'],
        "model_clients": [
            # {"model_name": "'gpt-4o-mini'", "model_type": "openai"},
            # {"model_name": "o3", "model_type": "openai"},
            {"model_name": "gpt-5-mini", "model_type": "openai"},
            # {"model_name": "claude-sonnet-4-20250514", "model_type": "anthropic"}
                          ],
        "n_agents": 20,

        "tool_parameters": {
            "sector_forecast": {
                "effort_thresholds": {"high": 5.0, "medium": 2.0},
                "effort_level_quality_mapping": {"high": 0.9, "medium": 0.7, "low": 0.5},
                "default_num_regimes": 2,
                "base_forecast_quality": 0.6,
                "default_regime_persistence": 0.8
            },
            "analyze_historical_performance": {
                "effort_thresholds": {"high": 3.0, "medium": 1.5},
                "high_effort_max_trades": 100,
                "high_effort_noise_factor": 0.05,
                "medium_effort_max_trades": 50,
                "medium_effort_noise_factor": 0.1,
                "low_effort_max_trades": 20,
                "low_effort_noise_factor": 0.2
            },
            "analyze_buyer_preferences": {
                "effort_thresholds": {"high": 3.0, "medium": 1.5},
                "high_effort_num_buyers": 30,
                "high_effort_num_test_offers": 12,
                "high_effort_analyze_by_attribute": True,
                "medium_effort_num_buyers": 15,
                "medium_effort_num_test_offers": 6,
                "medium_effort_analyze_by_attribute": False,
                "low_effort_num_buyers": 5,
                "low_effort_num_test_offers": 3,
                "low_effort_analyze_by_attribute": False
            },
            "research_competitive_pricing": {
                "effort_thresholds": {"high": 2.5, "medium": 1.2},
                "high_effort_num_buyers": 30,
                "high_effort_price_points": 12,
                "high_effort_lookback_trades": 50,
                "medium_effort_num_buyers": 15,
                "medium_effort_price_points": 6,
                "medium_effort_lookback_trades": 20,
                "low_effort_num_buyers": 8,
                "low_effort_price_points": 4,
                "low_effort_lookback_trades": 10
            },    
        },
        "max_chat_turns": 10,
        "max_chat_turns_single_agent": 10,
        "historical_periods": 20,
        "historical_production_cost": 30.0,  # Per forecast - ensures healthy margins
        "market_config": market_for_finance.MarketConfig(),
        "prompt_variant": "subtle"  # Options: "subtle" or "direct"
    }
    
    market_state = create_market_state()
    
    # Generate simple historical market data for agents to research
    generate_simple_historical_trades(market_state, config)
    
    # Visualize historical market patterns
    plot_historical_market_trends(market_state.all_trades)
    
    model = MarketModel(
      id=1,
      num_rounds=5,
      name="marketing_demo",
      agents=[],
      agent_metadata={},
      state=market_state,
      step=run_model_step,
      collect_stats=collect_stats_demo)
    
    agents, agent_metadata = create_agents(model, config)
    model.agents = agents
    model.agent_metadata = agent_metadata
    
    logger.info("About to create AI chats")
    chats = create_ai_chats(model, config)
    model.chats = chats
    logger.info(f"Created {len(model.chats)} chats")

    # Run async simulation loop (replacement for abm.run)
    logger.info(f"Starting simulation with {len(model.agents)} agents.")
    
    model.collect_stats(model, config)
    for _ in range(model.num_rounds):
        logger.info(f"Round {model.tick} begins.")
        await model.step(model, config)
        model.collect_stats(model, config)
        logger.info(f"Round {model.tick} ends.")
        logger.info("-" * 50)

    print("=== Simulation Summary ===")
    
    # Print tool call summary
    if tool_tracker:
        tool_tracker.print_summary()
        tool_tracker.print_failed_calls()
    
    model_results = model.model_results
    agent_results = model.agent_results
    
    print("=== Simulation completed. ===")
    
    # Plot trend over time in total trades executed by quality type
    print("Plotting trade trends by quality type...")
    
    import matplotlib.pyplot as plt
    
    quality_types = ['high_quality', 'medium_quality', 'low_quality']
    # To convert model_results to a DataFrame, we need to flatten the nested
    # dictionaries using pd.json_normalize
    import pandas as pd
    plot_df = pd.json_normalize(model_results, sep='_')
    plt.figure(figsize=(10, 6))
    for qtype in quality_types:
        trade_counts = plot_df[f"trade_counts_{qtype}"]
        plt.plot(plot_df.index + 1, trade_counts, label=qtype)
    plt.xlabel("Period")
    plt.ylabel("Number of Trades Executed")
    plt.title("Trades Executed Over Time by Quality Type")
    plt.legend()
    plt.show()


def run_demo_simulation():
    """Synchronous wrapper for the async simulation."""
    asyncio.run(run_demo_simulation_async())

if __name__ == "__main__":
    run_demo_simulation()