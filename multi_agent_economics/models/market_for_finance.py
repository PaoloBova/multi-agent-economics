"""
Market for Finance Model

This module implements a market for finance model, simulating the dynamics of
supply and demand, information flow, and budget allocation among agents.
"""

import numpy
from pydantic import Field, BaseModel

import numpy as np

def renormalize(probs):
    """
    Renormalize a 1D array of non-negative probabilities so they sum to 1.
    Zeros remain zeros.
    """
    probs = numpy.array(probs, dtype=float)
    total = probs.sum()
    if total > 0:
        return probs / total
    else:
        # If all zero, just return unchanged (or raise an error)
        return probs

def categorical_draw(probs):
    """
    Draw a single index j ∼ Categorical(probs).
    `probs` must sum to 1 (up to floating-point).
    """
    probs = numpy.array(probs, dtype=float)
    # Cumulative sums
    cum = numpy.cumsum(probs)
    u = numpy.random.rand()
    # find first index where cum ≥ u
    return int(numpy.searchsorted(cum, u, side='right'))

def sample_cart_multidraw(offers, probs, budget, T=10000):
    """
    Build a “cart” by sampling offers without replacement until budget runs out
    or no probability mass remains.
    
    offers: list of objects with a .price attribute
    probs:  list of choice probabilities p_j  (will be renormalized internally)
    budget: total spend limit
    T:      maximum number of draws (optional; if None, infinite)
    """
    probs = renormalize(probs)
    cart = []
    remaining = budget
    draws = 0

    while remaining > 0 and probs.sum() > 0 and (T is None or draws < T):
        j = categorical_draw(probs)
        price = offers[j].price
        if price <= remaining:
            cart.append(offers[j])
            remaining -= price
            # zero‐out that offer to avoid repeats
            probs[j] = 0
            probs = renormalize(probs)
        # even if the item didn't fit, we count the draw
        draws += 1

    return cart

def solve_continuous_knapsack(U, p, B):
    # U, p are arrays of same length; B is budget
    # find λ>0 s.t. sum_j min(1, max(0, U[j]/λ)) * p[j] == B
    λ_low, λ_high = 1e-9, max(U)/1e-9
    for _ in range(50):
        λ = 0.5*(λ_low + λ_high)
        spend = sum(min(1, max(0, U[j]/λ)) * p[j] for j in range(len(U)))
        if spend > B:
            λ_low = λ
        else:
            λ_high = λ
    x = [min(1, max(0, U[j]/λ)) for j in range(len(U))]
    return x

def greedy_budget_choice(offers, V, B):
    """
    offers: list of Offer
    V[j]: buyer’s gross valuation for offer j
    B: initial budget
    """
    scores = [ ( (V[j]-offers[j].price)/offers[j].price, j )
               for j in range(len(offers)) ]
    # stable‐sort descending, but break exact ties by a coin flip
    scores.sort(key=lambda x: (x[0], numpy.random.random()), reverse=True)

    cart = []
    remaining = B
    for _, j in scores:
        price = offers[j].price
        net   = V[j] - price
        if net > 0 and price <= remaining:
            cart.append(offers[j])
            remaining -= price
    return cart

def choice_model(buyer_state, offers, config):
    """Dispatches to the appropriate choice model based on configuration."""
    # Calculate value of each offer to the buyer
    V = lambda offer: numpy.dot(buyer_state.attr_weights, offer.attr_vector)
    value_of = [V(offer) for offer in offers]
    buyer_state.value_of = value_of  # Store for later use
    if "cart_draws" not in config:
        config.cart_draws = None  # Default to no limit

    if config.choice_model == "greedy":
        choices = greedy_budget_choice(offers, value_of, buyer_state.budget)
    elif config.choice_model == "logit_cart":
        choice_probs = numpy.exp(value_of) / numpy.sum(numpy.exp(value_of))
        choices = sample_cart_multidraw(offers, choice_probs, buyer_state.budget, T=config.cart_draws)
    elif config.choice_model == "knapsack":
        p = [o.price for o in offers]
        fractional_choices = solve_continuous_knapsack(value_of, p, buyer_state.budget)
        # Use sample_cart_multidraw to convert factional choices to discrete choices
        choices = sample_cart_multidraw(offers, fractional_choices, buyer_state.budget, T=config.cart_draws)
    else:
        raise ValueError(f"Unknown choice model: {config.choice_model}")
    return choices

def clear_market(choices, _model, config):
    """
    choices: list of (buyer_id, Offer) pairs
    Since goods are non-rival, every choice is filled.
    Transform into trades = [(buyer_id, org_id, price, qty=1, good_id), …]
    """
    trades = []
    if config.matching_rule == "non-rival":
        for buyer_id, offer in choices:
            trades.append((buyer_id, offer.org_id, offer.price, offer.quantity, offer.good_id))
    else:
        raise ValueError(f"Unknown matching rule: {config.matching_rule}")
    return trades

def resolve_ex_post_valuations(trades, model, market_cfg):
    """
    Resolve ex-post valuations for trades.
    This is where we would apply any additional logic to determine the
    actual value of trades based on market conditions or other factors.
    """
    for trade in trades:
        buyer_id, org_id, price, qty, good_id = trade
        # Assuming we have a function to get the valuation based on the good_id
        valuation = get_valuation(buyer_id, good_id)
        model.state.product_impact[good_id][buyer_id] = valuation

def compute_surpluses(trades, model, market_cfg):
    """Apply transfers and record surplus."""
    for buyer_state in model.state.buyers_state:
        buyer_surplus = 0.0
        for trade in trades:
            # Assuming trade is a tuple (buyer_id, org_id, price, qty, good_id)
            if trade[0] == buyer_state.buyer_id:
                good_id = trade[4]
                buyer_id = trade[0]
                # Add the previously calculated impact of the good on the buyer
                buyer_surplus += model.state.product_impact[good_id][buyer_id]
                buyer_surplus -= trade[2] * trade[3]  # price * qty
        buyer_state.surplus = buyer_surplus
    for seller_state in model.state.sellers_state:
        seller_surplus = 0.0
        for trade in trades:
            # Assuming trade is a tuple (buyer_id, org_id, price, qty, good_id)
            org_id = trade[1]
            if org_id == seller_state.org_id:
                seller_surplus += trade[2] * trade[3]  # price * qty
        seller_state.surplus = seller_surplus
        # Take care of any production costs from the seller
        # Assuming production_cost is defined in seller state
        seller_state.surplus -= seller_state.production_cost
        seller_state.total_profits += seller_state.surplus

def update_buyer_preferences(buyer, trades, obs_noise_var=1.0):
    """
    buyer.attr_mu[j]: prior mean of weight on attr j
    buyer.attr_sigma2[j]: prior variance of weight on attr j
    trades: list of (offer, price, surplus) that buyer executed this round
    """
    for offer, price, surplus in trades:
        x = offer.attr_vector       # e.g. [q_theme1, q_methodA, …]
        # we assume surplus ≈ x·β + noise, noise~N(0,obs_noise_var)
        for j, x_j in enumerate(x):
            # prior precision τ0 and likelihood precision τL
            τ0 = 1 / buyer.attr_sigma2[j]
            τL = x_j**2 / obs_noise_var
            # posterior precision & mean
            τ_post = τ0 + τL
            μ_post = (τ0*buyer.attr_mu[j] + x_j*surplus/obs_noise_var) / τ_post

            buyer.attr_sigma2[j] = 1 / τ_post
            buyer.attr_mu[j]     = μ_post

def transition_demand(model, market_cfg):
    # 1) For each buyer agent, collect their trades and update μ,σ²
    for b in filter_buyers(model.agents, 0):
        buyer_trades = [t for t in model.last_trades if t.buyer is b]
        update_buyer_preferences(b, [(t.offer, t.price, t.surplus) for t in buyer_trades])
    # 2) Rebuild any aggregate stats, e.g. population mean β
    all_mus = numpy.array([b.attr_mu for b in model.agents if b.is_buyer])
    model.state.aggregate_beta = all_mus.mean(axis=0)
    return model.state.demand_profile  # or store the aggregate in state

def run_market_dynamics(model, market_cfg):
    """Market dynamics: match supply and demand, clear trades"""
    offers = model.state.offers
    
    # 1. Each buyer runs choice model to select offers:

    buyers_state = model.state.buyers_state
    choices = []
    for buyer_state in buyers_state:
        offers_chosen = choice_model(buyer_state, offers, market_cfg)
        for offer in offers_chosen:
            choices.append((buyer_state.buyer_id, offer))
    
    # 2. Clearing mechanism: e.g., discrete double auction or posted‑price matching
    trades = clear_market(choices, model, market_cfg)
    model.state.all_trades.extend(trades)
    model.state.trades = trades  # Current round trades
    
    # 3. Resolve ex-post valuations and surpluses
    resolve_ex_post_valuations(trades, model, market_cfg)
    for trade in trades:
        compute_surpluses(trade, model, market_cfg)

    # 4. Demand-side shock for next round
    model.state.demand_profile = transition_demand(
        model.state.demand_profile,
        market_cfg.demand_shock_dist
    )

    return trades


def run_information_dynamics(model, info_cfg):
    "Information dynamics: update fundamentals, resolve predictions, inject news."
    # a) Evolve underlying fundamentals: e.g., geometric Brownian motion
    model.state.index_values = update_indices(model.state.index_values, info_cfg)

    # b) Resolve predictions scheduled for this tick (with lag L):
    for prediction in predictions_due(model.state):
        log_resolution(model.state, prediction)
        score = proper_scoring_rule(prediction.outcome, prediction.forecast)
        record_accuracy(model.state, prediction.agent_id, score)

    # c) Inject stochastic news events: public or private
    news_events = sample_news_events(state.agents, info_cfg.news_rates)
    for event in news_events:
        deliver_news(event) # Also decides who gets the news

def allocate_budgets(model, alloc_cfg):
    "Organizational rebalancing: principals allocate budgets across teams"
    scores = aggregate_scores(model.state, alloc_cfg)
    weights = alloc_cfg.pivot_function(scores)
    redistribute_budget(model.state, weights)

def model_step(model, config):
    "Top‑level model step, invoked once per tick (round)"
    run_market_dynamics(model, config.market_params)
    run_information_dynamics(model, config.info_params)
    allocate_budgets(model, config.budget_params)

def collect_stats(model):
    "Collect statistics from the model state"
    stats = {
        "total_trades": len(model.state.trades),
        "average_price": sum(trade.price for trade in model.state.trades) / len(model.state.trades) if model.state.trades else 0,
        "total_demand": sum(model.state.demand_profile.values()),
        "total_supply": sum(model.state.supply_profile.values())
    }
    return stats

class Offer(BaseModel):
    """ Represents an offer in the market."""
    good_id: str = Field(..., description="Unique identifier for the good")
    price: float = Field(..., description="Price of the offer")
    seller: str = Field(..., description="ID of the seller agent")
    attr_vector: list[float] = Field(..., description="Attributes of the offer (e.g., quality, features)")

class MarketState(BaseModel):
    """ Represents the state of a market in the simulation framework."""
    prices: dict = Field(..., description="Current prices of assets in the market")
    offers: list[Offer] = Field(..., description="List of offers available in the market")
    trades: list = Field(..., description="List of trades executed in the market")
    demand_profile: dict = Field(..., description="Demand profile for the market")
    supply_profile: dict = Field(..., description="Supply profile for the market")
    index_values: dict = Field(..., description="Values of market indices")

class MarketModel(BaseModel):
    """ Represents a market model in the simulation framework."""
    id: int = Field(..., description="Unique identifier for the market")
    name: str = Field(..., description="Name of the market")
    agents: list = Field(..., description="List of agents participating in the market")
    state: MarketState = Field(..., description="Current state of the market, including prices, trades, and other relevant data")
    step: callable = Field(..., description="Function to execute a step in the market model")
    collect_stats: callable = Field(..., description="Function to collect statistics from the market model")