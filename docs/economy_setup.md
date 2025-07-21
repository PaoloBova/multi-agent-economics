
# Multi-Agent LLM Product Market Simulation with AutoGen v0.4

## Overview and Setup

This solution uses **Microsoft's AutoGen (v0.4)** framework to orchestrate a multi-agent Large Language Model (LLM) simulation of a product market. AutoGen v0.4 is the latest stable version and supports **asynchronous multi-agent conversations**, integrated tool use, and state persistence. We will create a team of five agents (Principal, R\&D, Marketing, Pricing, Finance) that collaborate in a simulated business cycle. Each agent has domain-specific knowledge and AutoGen-registered tools to perform actions (budgeting, R\&D investment, marketing campaigns, price adjustments, etc.). The agents communicate via a **RoundRobinGroupChat** team so they share context and take turns automatically.

**Dependencies:** Ensure Python 3.10+ is installed. Using Poetry, add the AutoGen packages as dependencies (or install via pip). For example, the AutoGen docs recommend:

```bash
poetry add autogen-agentchat autogen-ext[openai]
# or via pip:
pip install autogen-agentchat autogen-ext[openai]
```

This installs the core AgentChat framework and OpenAI extension. An OpenAI API key should be set (e.g., in the `OPENAI_API_KEY` environment variable) to use GPT models. The code below is framework-agnostic and can work with OpenAI GPT-4/GPT-3.5 or other LLMs supported by AutoGen (including Azure OpenAI or local models via compatible APIs).

## Agent Roles and Tools

We design each agent with a clear role and assign them relevant **tools** for actions in the simulation. In AutoGen, tools are simply Python functions (with type hints and docstrings) that an agent can call during conversation. AutoGen v0.4 makes it easy to use tools: we can pass a list of tool functions when creating an AssistantAgent, and the agent will be able to invoke those functions as needed (no separate tool-executor agent is required in v0.4).

The agents and their responsibilities in each cycle are:

* **Principal Agent** – Coordinates the team. It may gather market feedback and, at the end of the cycle, calls the market update tool to finalize results. *(Tools: `conduct_focus_group`, `market_update_tool`)*
* **R\&D Agent** – Focuses on product development. It invests in R\&D to improve product quality. *(Tool: `invest_in_rd`)*
* **Marketing Agent** – Handles market outreach. It launches marketing campaigns to increase market share, and can conduct focus groups to gauge customer sentiment. *(Tools: `launch_campaign`, `conduct_focus_group`)*
* **Pricing Agent** – Sets the product price. It can adjust the price to optimize revenue or competitiveness. *(Tool: `adjust_price`)*
* **Finance Agent** – Manages the budget. It allocates available budget to R\&D and Marketing each cycle. *(Tool: `budget_tool`)*

Each tool function updates a shared **simulation state** (a Python dict) that represents the market and firm metrics. The state includes variables like current budget, market share, product quality, price, etc. A simulation **core** (the `market_update_tool`) uses these variables to compute outcomes (units sold, revenue, profit) and update the state for the next cycle. For realism, our market update considers a simple demand model and cost structure (e.g. it assumes a fixed production cost per unit).

**Memory Persistence:** The agents maintain context across cycles. We reuse the same agent instances in a continuous conversation, so each agent remembers past actions and outcomes by virtue of the shared message history. AutoGen allows saving and loading agent or team state if needed for long-term persistence, but in this implementation the state is kept in-memory by not resetting the team between cycles. This means agents can refer back to previous cycles’ decisions and results.

## Multi-Agent Simulation Workflow

We orchestrate the conversation in **cycles** corresponding to periods in the market (e.g., quarterly decision rounds). In each cycle, the process is:

1. **Broadcast State:** We inject a user message describing the current market state (budget, market conditions, etc.) and instruct the team to begin the decision-making for that cycle. This acts as the initial prompt for that round.
2. **Agent Turns (Decision Phase):** The RoundRobinGroupChat will let agents speak in sequence. The Principal agent (as coordinator) goes first, possibly analyzing state or gathering feedback (e.g., via a focus group tool). Next, Finance allocates the budget via the `budget_tool`. Then R\&D, Marketing, and Pricing agents each take their turns, proposing and executing actions using their tools (`invest_in_rd`, `launch_campaign`, `adjust_price`). Each tool call is executed immediately and its result is inserted into the chat – since we registered tools with the agents, the LLM will generate a function call and AutoGen will run the function, returning the result for the agent to use. We set `reflect_on_tool_use=True` for agents so they incorporate tool results into their messages (instead of just returning raw data).
3. **Market Update and Termination:** After the functional agents act, the Principal agent gets the final turn in the cycle. The Principal calls the `market_update_tool` to update the market state based on all actions (sales and profit are calculated and the state dict is updated). The tool returns a summary of outcomes, which the Principal shares. The Principal then ends its message with a special token (e.g., "CYCLE\_COMPLETE"). We use a **TextMentionTermination** condition to detect this token and terminate the group chat for that cycle. (We also combine a max message limit as a safety net.) Once the cycle ends, the updated state is retained for the next iteration. The loop then proceeds to the next cycle with a new user/state message.

Using the **Console UI** streaming, we can observe the multi-agent dialogue in real time, which is helpful for debugging and understanding the agents’ collaboration. Below is the complete Python implementation. It defines the tools and agents, then runs a multi-cycle simulation.

## Complete Implementation

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# --- Simulation State ---
# Global state dictionary to hold market and firm parameters.
market_state = {
    "period": 0,
    "budget": 100,              # available budget for the cycle
    "market_share": 0.20,       # current market share of the product (20% initially)
    "product_quality": 5.0,     # product quality level (arbitrary scale, 5/10 initially)
    "price": 100.0,             # current product price
    "unit_cost": 50.0,          # cost to produce one unit (for profit calculation)
    "cumulative_profit": 0.0,   # total profit accumulated
    "last_profit": 0.0,         # profit in the last cycle
    "rd_budget_alloc": 0,       # budget allocated to R&D this cycle
    "marketing_budget_alloc": 0,# budget allocated to Marketing this cycle
    "rd_spend_used": 0,         # actual R&D spend used this cycle
    "marketing_spend_used": 0   # actual Marketing spend used this cycle
}

# --- Tool Definitions ---
def budget_tool(rd_budget: int, marketing_budget: int) -> str:
    """Allocate budget (int) for R&D and Marketing for this cycle."""
    total = rd_budget + marketing_budget
    # Ensure allocations do not exceed available budget
    if total > market_state["budget"]:
        available = market_state["budget"]
        # Scale down allocations proportionally to fit available budget
        if total > 0:
            scale = available / total
            rd_alloc = int(rd_budget * scale)
            mk_alloc = int(marketing_budget * scale)
            # Adjust rounding remainder if any
            if rd_alloc + mk_alloc < available:
                mk_alloc = available - rd_alloc
        else:
            rd_alloc = mk_alloc = 0
        market_state["rd_budget_alloc"] = rd_alloc
        market_state["marketing_budget_alloc"] = mk_alloc
        return (f"Only ${available} available. Adjusted allocation: R&D = ${rd_alloc}, Marketing = ${mk_alloc}.")
    else:
        market_state["rd_budget_alloc"] = rd_budget
        market_state["marketing_budget_alloc"] = marketing_budget
        return (f"Budget allocated: R&D = ${rd_budget}, Marketing = ${marketing_budget}.")

def invest_in_rd(amount: int) -> str:
    """Invest in R&D with the given amount to improve product quality."""
    # Cap the investment to the allocated R&D budget
    alloc = market_state.get("rd_budget_alloc", market_state["budget"])
    spend = min(amount, alloc)
    market_state["rd_spend_used"] = spend
    # Improve product quality (e.g., 1 quality point per $10 spent)
    improvement = spend / 10.0
    market_state["product_quality"] += improvement
    return (f"Invested ${spend} in R&D. Product quality increased by {improvement:.1f} to {market_state['product_quality']:.1f}.")

def launch_campaign(amount: int) -> str:
    """Launch a marketing campaign with the given budget to boost market share."""
    alloc = market_state.get("marketing_budget_alloc", market_state["budget"])
    spend = min(amount, alloc)
    market_state["marketing_spend_used"] = spend
    # Increase market share (e.g., 0.1% per $1 spent, scaled down)
    gain = spend / 1000.0  # $1000 -> 1% (0.01) increase
    old_share = market_state["market_share"]
    new_share = old_share + gain
    # Cap market share below 100%
    market_state["market_share"] = new_share if new_share < 0.99 else 0.99
    percent_gain = (market_state["market_share"] - old_share) * 100
    return (f"Launched campaign with ${spend}. Market share +{percent_gain:.2f}% to {market_state['market_share']*100:.2f}%.")

def adjust_price(new_price: float) -> str:
    """Adjust the product price to a new value."""
    old_price = market_state["price"]
    market_state["price"] = new_price
    # Apply a simple price elasticity effect on market share:
    if old_price > 0:
        change_pct = (new_price - old_price) / old_price  # relative price change
    else:
        change_pct = 0.0
    # Assume -0.5 relative share change for +100% price change (linear approximation)
    share_change = -0.5 * change_pct * market_state["market_share"]
    # Update market share (cannot go below 0 or above 0.99)
    market_state["market_share"] = max(0.0, min(0.99, market_state["market_share"] + share_change))
    return (f"Price changed from ${old_price:.2f} to ${new_price:.2f}. Updated market share: {market_state['market_share']*100:.2f}%.")

def conduct_focus_group() -> str:
    """Conduct a focus group to get customer feedback on the product."""
    # Generate a simple feedback summary based on current quality and price
    q = market_state["product_quality"]
    p = market_state["price"]
    feedback_points = []
    if q >= 8:
        feedback_points.append("customers love the product quality")
    elif q <= 3:
        feedback_points.append("customers are unhappy with the product quality")
    if p > 120:
        feedback_points.append("many customers feel the price is too high")
    elif p < 80:
        feedback_points.append("customers find the price very reasonable")
    if not feedback_points:
        feedback_points.append("customers feel the product is satisfactory for its price")
    feedback = " and ".join(feedback_points)
    return f"Focus group feedback: {feedback}."

def market_update_tool() -> str:
    """Update the market state at end of cycle: compute sales, revenue, profit, etc."""
    market_state["period"] += 1  # advance to next period
    # Simple demand model: sell units proportional to market share
    market_size = 1000  # total market potential (e.g., 1000 units)
    units_sold = int(market_state["market_share"] * market_size)
    # Calculate revenue and profit
    revenue = units_sold * market_state["price"]
    production_cost = units_sold * market_state.get("unit_cost", 0)
    total_cost = production_cost + market_state.get("rd_spend_used", 0) + market_state.get("marketing_spend_used", 0)
    profit = revenue - total_cost
    # Update financial metrics
    market_state["last_profit"] = profit
    market_state["cumulative_profit"] += profit
    # Update available budget for next cycle (reinvest profits, don't allow negative budget)
    new_budget = market_state["budget"] + profit
    market_state["budget"] = new_budget if new_budget > 0 else 0
    # Prepare summary of outcomes
    summary = (f"Sold {units_sold} units @ ${market_state['price']} each. "
               f"Revenue=${revenue:.0f}, Cost=${total_cost:.0f}, Profit=${profit:.0f}. "
               f"Budget now=${market_state['budget']:.0f}.")
    return summary

# --- Agent and Team Initialization ---
async def main():
    # Initialize OpenAI model client (using GPT-4 here; substitute "gpt-3.5-turbo" if desired)
    model_client = OpenAIChatCompletionClient(model="gpt-4")
    
    # Create agents with roles, system prompts, and tools
    principal = AssistantAgent(
        name="Principal",
        model_client=model_client,
        system_message=(
            "You are the Principal (team leader) of a company. You coordinate the R&D, Marketing, Pricing, and Finance agents. "
            "At the start of each cycle, you assess the market state (provided by the user) and may conduct a focus group for customer feedback. "
            "Assign tasks or guidance to the team. After the others take actions, use the market_update_tool to finalize the cycle's outcome. "
            "End your turn with 'CYCLE_COMPLETE' once the market update is done."
        ),
        tools=[conduct_focus_group, market_update_tool],
        reflect_on_tool_use=True
    )
    finance = AssistantAgent(
        name="Finance",
        model_client=model_client,
        system_message=(
            "You are the Finance Manager. Allocate the available budget between R&D and Marketing each cycle. "
            "Ensure the allocations do not exceed the total budget. Provide reasoning if needed, then use the budget_tool with your allocations."
        ),
        tools=[budget_tool],
        reflect_on_tool_use=True
    )
    rd_agent = AssistantAgent(
        name="R&D",
        model_client=model_client,
        system_message=(
            "You are the R&D Manager. Your goal is to improve the product quality by investing in research and development. "
            "Use the budget allocated to R&D wisely (do not exceed it). Provide a brief plan, then call invest_in_rd to spend the R&D budget."
        ),
        tools=[invest_in_rd],
        reflect_on_tool_use=True
    )
    marketing = AssistantAgent(
        name="Marketing",
        model_client=model_client,
        system_message=(
            "You are the Marketing Manager. You aim to increase the product's market share and customer interest. "
            "Use your marketing budget for campaigns (do not exceed the allocated amount). You can also conduct a focus group if needed to gauge customer sentiment. "
            "Describe your plan briefly and call launch_campaign to execute it."
        ),
        tools=[launch_campaign, conduct_focus_group],
        reflect_on_tool_use=True
    )
    pricing = AssistantAgent(
        name="Pricing",
        model_client=model_client,
        system_message=(
            "You are the Pricing Manager. Decide on the product's price for the upcoming period to maximize revenue and market share. "
            "Consider the market conditions (competition, demand elasticity). Announce the new price and use adjust_price to apply it."
        ),
        tools=[adjust_price],
        reflect_on_tool_use=True
    )
    
    # Set termination when the Principal signals cycle completion (or if too many messages).
    termination_condition = TextMentionTermination("CYCLE_COMPLETE") | MaxMessageTermination(max_messages=30)
    
    # Create the multi-agent team with round-robin turn-taking
    team = RoundRobinGroupChat(
        agents=[principal, finance, rd_agent, marketing, pricing],
        termination_condition=termination_condition
    )
    
    # Run multiple cycles of the simulation
    num_cycles = 3  # e.g., simulate 3 cycles (you can adjust as needed)
    for cycle in range(1, num_cycles + 1):
        # Compose the user prompt describing the current state for this cycle
        state_msg = (f"**[Cycle {cycle} State Update]**\n"
                     f"Budget available: ${market_state['budget']:.0f}\n"
                     f"Last cycle profit: ${market_state['last_profit']:.0f}\n"
                     f"Product quality: {market_state['product_quality']:.1f}\n"
                     f"Market share: {market_state['market_share']*100:.1f}%\n"
                     f"Price: ${market_state['price']:.2f}\n"
                     "Team, please discuss and make decisions for this cycle.")
        print(f"\n========== Cycle {cycle} ==========")
        # Run the group chat for the cycle, streaming the conversation to console
        try:
            async for msg in team.run_stream(task=state_msg):
                # The Console will format and display each message (source, content, etc.)
                await Console([msg])
        finally:
            # After the cycle completes (termination reached), print the updated state summary
            outcome = (f"End of Cycle {cycle}: Profit=${market_state['last_profit']:.0f}, "
                       f"Cumulative Profit=${market_state['cumulative_profit']:.0f}, "
                       f"New Budget=${market_state['budget']:.0f}, Market Share={market_state['market_share']*100:.1f}%, "
                       f"Product Quality={market_state['product_quality']:.1f}")
            print(outcome)
    
    await model_client.close()

# Run the simulation
asyncio.run(main())
```

**How it works:** The code first defines the global `market_state` and all six tool functions. Each tool returns a textual summary of its action for the agent to incorporate into the conversation. We then create an `AssistantAgent` for each role, providing a descriptive system prompt and attaching the relevant tools (via the `tools=[...]` parameter). All agents use the same `OpenAIChatCompletionClient` (here configured for GPT-4) to generate responses. We combine a text termination condition on `"CYCLE_COMPLETE"` with a max-messages cap to ensure the chat ends after each cycle.

We set up a `RoundRobinGroupChat` with the five agents. In the main loop, for each cycle we formulate a **state update message** that includes key state variables (budget, profit, quality, etc.) as the task prompt. This simulates a "broadcast" of the current situation to the team. We then call `team.run_stream(...)` to start the conversation for that cycle and use `Console` to stream messages to the console in real time. As the agents respond in turn, they will call their tools to perform actions. For example, the Finance agent will use `budget_tool` to set R\&D and marketing budgets, the R\&D agent will call `invest_in_rd` to improve quality, and so on. AutoGen handles the function call execution behind the scenes and inserts the results into the chat for agents to see. Finally, the Principal agent calls `market_update_tool`, which computes sales, profit, and updates the shared state. The Principal's message containing the market outcome ends with "CYCLE\_COMPLETE", triggering the termination of the round. We then print a brief summary of the cycle's results from the updated state. The loop continues to the next cycle, carrying over the state (note that `market_state['budget']` has been updated with new funds or losses, and agents retain memory of what happened in prior cycles).

## Example Outcome

When you run this program, you will see a multi-turn conversation for each cycle. Each agent will contribute in order, for example:

* **Principal:** (sees state, possibly runs a focus group) shares customer feedback and outlines strategy.
* **Finance:** decides allocations and calls `budget_tool` (result e.g. "Budget allocated: R\&D=\$X, Marketing=\$Y").
* **R\&D:** proposes an R\&D plan and calls `invest_in_rd` (e.g. "Invested \$X in R\&D, quality increased to ...").
* **Marketing:** outlines a campaign and calls `launch_campaign` (e.g. "Launched campaign with \$Y, market share now ...").
* **Pricing:** sets a new price via `adjust_price` (e.g. "Price changed from ... to ..., market share now ...").
* **Principal:** calls `market_update_tool` to compute outcome (e.g. "Sold N units... Profit=\$..., Budget now=\$... CYCLE\_COMPLETE").

After each cycle, the printed summary might show evolving metrics (e.g., increasing cumulative profit, changing market share, etc.), demonstrating the **persistent state and memory** across cycles. This implementation is ready to run and can be further extended (e.g., adding more complex market dynamics, additional agents, or integrating a GUI via AutoGen Studio). The use of AutoGen's multi-agent framework ensures that the agents' collaboration and tool usage are handled in a robust, conversational workflow, illustrating a powerful pattern for agentic AI in a business simulation.
