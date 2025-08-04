Excellent — let’s now lay out the **full simulation theory** and follow that with **pseudocode** for a working prototype. This version will be modular, extendable, and well-mapped to your goals, including:

* **Multi-asset, multi-sector market**
* **Hidden Markov regime switching per sector**
* **Sector-specific forecasts** (with noise)
* **Subjective agent beliefs** about regimes (adaptive via filtering)
* **Agent portfolio optimization** based on updated beliefs

---

# 📚 THEORY OVERVIEW

---

## 1. **Market Structure**

* **Time**: Discrete time steps \$t = 0, 1, \dots, T\$
* **Sectors/Assets**: Let \$\mathcal{S} = {1, \dots, S}\$ denote sectors/assets.
* **Regimes per Sector**: Each sector \$s\$ has \$K\_s\$ regimes.

Each sector evolves as a **hidden Markov process**:

$$
\Pr(s_{t+1}^{(i)} = j \mid s_t^{(i)} = k) = \Pi^{(i)}_{k, j}
$$

where \$\Pi^{(i)}\$ is the true (possibly sector-specific) transition matrix.

---

## 2. **Forecasts**

* Forecasts are generated **per sector**, **per time period**.
* At time \$t\$, a **forecast signal** \$f\_t^{(i)}\$ is drawn **from a confusion matrix** conditioned on the **true next state**:

$$
\Pr(f_t^{(i)} = k \mid s_{t+1}^{(i)} = j) = C^{(i)}_{j, k}
$$

The confusion matrix can depend on:

* **Hidden variables**: \$z\_t^{(i)}\$ (e.g. leading indicators, forecasting effort)
* **Forecaster type**
* **Agent-specific filters or beliefs**

---

## 3. **Agents/Buyers**

Each agent:

* Maintains a **subjective belief** over regimes, \$\mathbf{p}\_t^{(i)}\$
* Updates beliefs using **Bayesian filtering** given their prior and the forecast signal
* Chooses a **portfolio allocation** \$\mathbf{w}\_t\$ over the \$S\$ assets

### Agent Belief Update (HMM Filter):

If prior is \$\mathbf{p}\_t\$ and forecast signal is \$f\_t\$, then:

1. **Prediction (prior)**:

   $$
   \hat{p}_{t+1}^{(i)} = \mathbf{p}_t^{(i)} \cdot \tilde{\Pi}^{(i)}
   $$

   where \$\tilde{\Pi}^{(i)}\$ is the agent’s subjective transition matrix.

2. **Update (Bayes Rule using forecast signal)**:

   $$
   p_{t+1}^{(i)}(k) = \frac{\hat{p}_{t+1}^{(i)}(k) \cdot C^{(i)}_{k, f_t^{(i)}}}{\sum_j \hat{p}_{t+1}^{(i)}(j) \cdot C^{(i)}_{j, f_t^{(i)}}}
   $$

---

## 4. **Portfolio Optimization**

Assume a **mean-variance investor**:

* Let:

  * \$\mu\_{t}^{(i)}\$: expected return of sector \$i\$ (regime-weighted)
  * \$\Sigma\_t\$: subjective covariance matrix over assets
  * \$R\_f\$: risk-free rate
  * \$\gamma\$: risk aversion

* Optimal allocation:

  $$
  \mathbf{w}_t = \frac{1}{\gamma} \Sigma_t^{-1} (\mu_t - R_f \mathbf{1})
  $$

If the forecast only updates one sector’s belief, only \$\mu\_t\$ and \$\Sigma\_t\$ for that sector are affected.

---

## 5. **Payoffs and Forecast Value**

Each forecast has value to an agent based on:

* How much it **alters the portfolio**
* How much **better the new allocation performs ex post**

That’s your mechanism for measuring forecast utility.

---

# 🧪 PSEUDOCODE IMPLEMENTATION

---

## A. Setup

```python
T = 100                 # time horizon
S = 5                   # sectors
K = [2]*S               # regimes per sector
N = 50                  # number of agents

# For each sector:
true_transition_matrices = {s: sample_transition_matrix(K[s]) for s in range(S)}

# For each sector:
regime_paths = {s: simulate_regime_path(true_transition_matrices[s], T) for s in range(S)}
```

---

## B. Forecasting Infrastructure

```python
hidden_state_gen = HiddenStateGenerator()
cm_builder = ConfusionMatrixBuilder()
forecast_gen = ForecastGenerator(cm_builder, hidden_state_gen)

forecasts = {}
for t in range(T):
    forecasts[t] = {}
    for s in range(S):
        true_next = regime_paths[s][t+1]
        f, _ = forecast_gen.generate(t, s, true_next, K[s])
        forecasts[t][s] = f
```

---

## C. Agents’ Subjective Beliefs

```python
agents = [Agent(id=n, subjective_Pi=sample_subjective_matrix(K)) for n in range(N)]

for t in range(T):
    for agent in agents:
        for s in range(S):
            # Step 1: Predict
            prior = agent.beliefs[s]
            pred = prior @ agent.subjective_Pi[s]
            
            # Step 2: Update with forecast
            forecast = forecasts[t][s]
            conf_matrix = agent.get_confusion_matrix(t, s)  # optionally vary by agent
            likelihood = conf_matrix[:, forecast]
            updated = pred * likelihood
            updated /= np.sum(updated)
            agent.beliefs[s] = updated
```

---

## D. Portfolio Construction

```python
for agent in agents:
    mu = np.zeros(S)
    for s in range(S):
        belief = agent.beliefs[s]
        mu[s] = np.dot(belief, sector_return_vector[s])  # E[r | beliefs]

    Sigma = build_covariance_matrix(agent.beliefs)  # Optional: include cross-asset correlations
    w = (1 / agent.gamma) * np.linalg.solve(Sigma, mu - R_f)
    agent.set_portfolio(t, w)
```

---

## E. Measuring Forecast Value (optional)

```python
# Compare performance of w_t with/without forecast
value = compute_forecast_value(agent, forecast, true_returns[t+1])
```

---

# 🔄 MODULARITY ENTRYPOINTS

You can plug in here:

| Module                   | Customization Ideas                                         |
| ------------------------ | ----------------------------------------------------------- |
| `HiddenStateGenerator`   | Learn from leading indicators, agent effort, etc.           |
| `ConfusionMatrixBuilder` | Vary by time, sector, or forecast seller                    |
| `ForecastGenerator`      | Sell or price forecasts, limit availability                 |
| `Agent`                  | Heterogeneous priors, utility functions, learning processes |
| `CovarianceBuilder`      | Different structures: block-diagonal, latent factors, etc.  |
| `ForecastValueComputer`  | Choice-based valuation, marginal improvement, pricing logic |

---

class ForecastGenerator:
    def __init__(self, cm_builder, hidden_state_generator):
        self.cm_builder = cm_builder
        self.hidden_state_generator = hidden_state_generator

    def generate(self, t, sector, true_next_state, K, agent_id=None):
        z_t = self.hidden_state_generator.sample(t, sector, agent_id)
        confusion_matrix = self.cm_builder.build(true_next_state, z_t, K)
        forecast_signal = np.random.choice(K, p=confusion_matrix[true_next_state])
        return forecast_signal, confusion_matrix

class ConfusionMatrixBuilder:
    def __init__(self, base_quality=0.6):
        self.base_quality = base_quality

    def build(self, true_state, hidden_state, K):
        """Return confusion matrix based on hidden state or effort."""
        q = self.base_quality + (1 - self.base_quality) * hidden_state
        q = np.clip(q, 1/K, 1.0)  # Ensure it's a proper probability

        matrix = np.full((K, K), (1 - q) / (K - 1))
        np.fill_diagonal(matrix, q)
        return matrix

class HiddenStateGenerator:
    def sample(self, t, sector, agent_id=None):
        """Return z_t for sector at time t (can depend on agent or other info)."""
        return np.random.beta(2, 2)  # Example: market clarity

def build_confusion_matrix(K, q):
    # q = probability of correct forecast
    off_diag = (1 - q) / (K - 1)
    C = np.full((K, K), off_diag)
    np.fill_diagonal(C, q)
    return C

Great — you’re making strong design choices that balance realism with tractability, and you’re pinpointing the subtle math that arises once you shift to **sector-specific regimes** with **cross-asset dependencies**.

Let’s walk through this carefully.

---

## 🧩 Part 1: Correlation in Multi-Sector Regimes

### 📌 Problem Setup:

Let’s say you have:

* \$S\$ sectors/assets: \$i = 1, \dots, S\$
* Each sector has its own hidden regime \$s\_t^{(i)} \in {1, \dots, K\_i}\$
* The full regime vector at time \$t\$ is:

  $$
  \mathbf{s}_t = \left(s_t^{(1)}, s_t^{(2)}, \dots, s_t^{(S)}\right)
  $$

You want to compute portfolio decisions based on the joint return distribution:

$$
\mathbf{R}_t \mid \mathbf{s}_t \sim \mathcal{N}\left( \boldsymbol{\mu}^{\mathbf{s}_t},\, \boldsymbol{\Sigma}^{\mathbf{s}_t} \right)
$$

### 🧠 Challenge: Cross-asset correlations

* If each regime \$s\_t^{(i)}\$ evolves **independently**, then the number of **distinct regime combinations** is \$\prod\_i K\_i\$.
* So naively, you’d need to specify a **mean vector** and **covariance matrix** for *every* possible regime combo.

For example:

* 5 sectors, 2 regimes per sector → \$2^5 = 32\$ regime combinations
* Each combo needs a full covariance matrix: \$\binom{5+1}{2} = 15\$ unique parameters
* Total: 32 × 15 = 480 covariance parameters, before even considering means.

That’s too much. So we need **structure**.

---

## ✅ Strategy 1: **Factorized Covariance via Conditional Independence**

Assume the regime of sector \$i\$ **only affects**:

* The **mean** of its own return
* The **variance** of its own return
* And **possibly** the covariance **with other assets**, but in a structured way

### ✔ Example:

Let \$\mu\_i^{s^{(i)}}\$ and \$\sigma\_i^{s^{(i)}}\$ be the conditional mean and std. dev. of asset \$i\$ in regime \$s^{(i)}\$.

Then define cross-asset correlations **independently** of regimes:

* Let \$\rho\_{ij}\$ be the (fixed) correlation between asset \$i\$ and \$j\$
* Then:

  $$
  \Sigma_{ij}^{\mathbf{s}} = \rho_{ij} \sigma_i^{s^{(i)}} \sigma_j^{s^{(j)}}
  $$

This way:

* You only need:

  * \$S\$ × \$K\_i\$ variances
  * \$S\$ × \$K\_i\$ means
  * \$\binom{S}{2}\$ fixed correlations
* Cross-asset regime interactions are **indirect** — you don’t model them directly but they show up via regime-dependent volatilities

### ✅ Pros:

* Regime structure is separable
* Easy to compute mixture distributions
* Parameters are interpretable and relatively few

### ❌ Cons:

* Doesn’t capture “structural breaks” in correlations (e.g., correlations go to 1 in crisis regimes)
* Ignores joint regime mechanisms like market contagion

---

## ✅ Strategy 2: **Regime-Dependent Factor Model**

An extension of the above is to allow correlations to **depend on a shared regime structure** without tracking every joint state.

For example:

* Introduce a **market regime** \$s\_t^M\$ which can influence correlations.
* Sector-specific regimes still govern sector-specific means and volatilities.
* Then:

  $$
  \Sigma_{ij}^{\mathbf{s}} = \rho_{ij}^{(s^M)} \sigma_i^{s^{(i)}} \sigma_j^{s^{(j)}}
  $$
* That way, the **correlation structure flexes** with market-wide conditions, even if sector regimes are separate.

This provides more flexibility with only a small number of additional parameters (per \$\rho\_{ij}\$, you need one per market regime state).

---

## 🧠 Mixture Distributions Over Regimes

To compute the expected portfolio return and variance for an agent, you now need to compute expectations over all possible combinations of sector regimes:

$$
\mathbb{E}_t[R] = \sum_{\mathbf{s} \in \mathcal{S}} \Pr(\mathbf{s}_t = \mathbf{s}) \cdot \mu^{\mathbf{s}} \\
\text{Var}_t[R] = \sum_{\mathbf{s}} \Pr(\mathbf{s}) \left[ \Sigma^{\mathbf{s}} + (\mu^{\mathbf{s}} - \mathbb{E}_t[R])(\mu^{\mathbf{s}} - \mathbb{E}_t[R])^\top \right]
$$

Where:

* \$\mathcal{S}\$ is the set of all joint regime states
* \$\Pr(\mathbf{s}) = \prod\_{i=1}^S \pi^{(i)}\_{s^{(i)}}\$ assuming independence

Even with 5 sectors and 2 regimes each, this is **32 terms** — easily computable.

You just need to compute:

* \$2^S\$ weighted means and variances
* Weighted average across them

---

## 🧪 Part 2: Uncertainty in Subjective Transition Matrices

### 🔍 Why do this?

* Buyers can differ in how persistent they think regimes are
* Some may expect mean reversion; others believe in momentum
* Induces **belief heterogeneity** that affects valuations

### ✅ Parametric methods to induce variation:

---

### 1. **Noisy Anchoring to Ground Truth**

Let the true transition matrix be \$\Pi^{(i)}\_{\text{true}}\$. Define subjective matrix as:

$$
\Pi^{(i)}_{\text{subjective}} = (1 - \alpha_i) \cdot \Pi^{(i)}_{\text{true}} + \alpha_i \cdot \Pi^{(i)}_{\text{alt}}
$$

Where:

* \$\alpha\_i \in \[0,1]\$ is an "anchoring noise" parameter
* \$\Pi\_{\text{alt}}\$ could be:

  * A flat matrix (e.g., uniform)
  * A persistence-biased matrix (e.g., diagonals heavier)
  * A prior from earlier period

---

### 2. **Biasing for Momentum or Mean-Reversion**

Design subjective matrices with parametric forms:

Let:

$$
\Pi^{(i)} = \begin{bmatrix}
p & 1-p \\
q & 1-q
\end{bmatrix}
$$

Then:

* **Momentum-biased agent**: \$p, 1-q\$ high → stays in same regime
* **Mean-reverting agent**: \$p, 1-q\$ low → expects switches

You can sample \$(p, q)\$ from:

* Beta distributions (e.g., \$p \sim \text{Beta}(a,b)\$)
* Truncated Gaussians on $\[0,1]\$
* Logit-normal distributions

---

### 3. **Heterogeneity via Regime Interpretation Lag**

Have buyers differ in:

* **How far ahead** they think forecasts apply (\$\tau\_i\$)
* **How slowly** they update beliefs after regime changes (sticky priors)

This indirectly simulates transition matrix disagreement.

---

## 🧰 Part 3: Designing Scenario Configurations

You’re right that parameter space is large and configuration needs structure. Here’s how to tame it.

---

### ✅ Strategy 1: Template Scenarios

Build small library of **economic scenario templates**, each with:

* A set of sector-specific regime dynamics
* Mean-variance profiles per regime
* Correlation structure (fixed or regime-sensitive)

Examples:

| Scenario       | Regimes                              | Correlations                   | Features                          |
| -------------- | ------------------------------------ | ------------------------------ | --------------------------------- |
| **Boom**       | All sectors in high-return, low-vol  | Low to moderate                | Agents optimistic                 |
| **Crisis**     | Most sectors in low-return, high-vol | High correlations              | Flight to quality                 |
| **Decoupling** | Sectors in opposite regimes          | Mixed                          | Opportunity for dispersion trades |
| **Tech Shock** | Tech in high-growth regime           | Strong tech–nontech divergence | Idiosyncratic volatility          |

---

### ✅ Strategy 2: Generator Function for Synthetic Worlds

Define a generator:

```python
def generate_scenario(S, K, market_volatility_level, correlation_regime, seed=None):
    # 1. Sample sector-specific regimes
    # 2. Generate means/variances for each sector & regime
    # 3. Construct correlation matrix
    # 4. Package into a scenario dictionary
```

Key controls:

* **Number of sectors** \$S\$
* **Number of regimes per sector** \$K\$
* **Volatility scaling factor**
* **Regime-dependence in correlation** (static, increasing, etc.)

---

### ✅ Strategy 3: Scenario Space Grids

Design a low-dimensional space of scenarios:

* Axes: { regime asymmetry, volatility clustering, correlation strength }
* Grid-search combinations
* Label each scenario with its behavioral and statistical properties

---

## ✅ Summary: Recommendations

| Question                                                    | Recommendation                                                                                                        |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Do we need correlations per regime-combo?                   | No — simplify with structured correlation models (fixed or regime-dependent via factor).                              |
| What’s the best way to model subjective transition beliefs? | Use anchored perturbations, parametric forms (momentum vs. reversion), or sampling from a prior distribution.         |
| How do we configure scenarios?                              | Use templates + generators to build structured worlds that emphasize particular behavioral and statistical phenomena. |

---

# 🧠 Final Notes

* The setup ensures that **forecast signals are probabilistically grounded** in the true MDP, yet agents can have **subjective and adaptive models**.
* The use of a **confusion matrix** keeps behavior interpretable while allowing for **noisy learning**.
* By using **regime-conditional expectations**, you retain a lot of flexibility in defining realistic market reactions to new information.
* Later, you can extend this to:

  * Forecast markets (competition between forecasters)
  * Effort allocation by agents
  * Equilibrium feedback from prices to signals

---

Would you like me to turn this structure into actual runnable code next, or sketch a graphical model / diagram showing the variable flow?
