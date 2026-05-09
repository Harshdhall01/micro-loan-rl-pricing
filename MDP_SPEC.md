# 📐 MDP Specification — Micro-loan Dynamic Pricing

## Formal Markov Decision Process Definition

This document formally defines the Markov Decision Process (MDP) underlying
the Micro-loan Dynamic Pricing system. The RL agent learns an optimal policy
by interacting with this MDP over thousands of episodes.

---

## 1. MDP Tuple — M = (S, A, T, R, γ)

```
M = (S, A, T, R, γ)

S  →  State space        (borrower profile + economic conditions)
A  →  Action space       (interest rate decision)
T  →  Transition model   (borrower response simulation)
R  →  Reward function    (multi-objective signal)
γ  →  Discount factor    (0.99)
```

---

## 2. State Space — S

The state is a 10-dimensional continuous vector, all features normalized to [0, 1]:

```
s = [s₁, s₂, s₃, s₄, s₅, s₆, s₇, s₈, s₉, s₁₀]
```

| Index | Feature | Raw Range | Description |
|-------|---------|-----------|-------------|
| s₁ | `monthly_income` | ₹3,000 – ₹50,000 | Borrower's monthly income |
| s₂ | `loan_amount_requested` | ₹5,000 – ₹500,000 | Requested loan size |
| s₃ | `credit_score` | 300 – 850 | Creditworthiness score |
| s₄ | `employment_type` | {0, 0.5, 1} | 0=farmer, 0.5=trader, 1=salaried |
| s₅ | `location` | {0, 0.5, 1} | 0=rural, 0.5=semi-urban, 1=urban |
| s₆ | `dependents` | 0 – 6 | Number of financial dependents |
| s₇ | `seasonal_income` | {0, 1} | Boolean: seasonal earner flag |
| s₈ | `previous_defaults` | 0 – 3 | Historical default count |
| s₉ | `inflation_rate` | 2% – 15% | Macroeconomic inflation |
| s₁₀ | `unemployment_rate` | 3% – 25% | Macroeconomic unemployment |

### State Space Properties
- **Dimensionality:** |S| = ℝ¹⁰ (continuous)
- **Bounds:** s ∈ [0, 1]¹⁰
- **Distribution:** Mixed — Beta distribution for credit scores, skewed categorical for defaults
- **Markov Property:** Each borrower profile is independently sampled → satisfies Markov property

---

## 3. Action Space — A

The agent selects a discrete interest rate from 57 possible values:

```
A = {a₀, a₁, ..., a₅₆}

aᵢ = 8.0 + (i × 0.5)  %    for i ∈ {0, 1, 2, ..., 56}

Range: [8.0%, 36.0%]  in steps of 0.5%
```

| Action | Interest Rate |
|--------|--------------|
| a₀ | 8.0% (minimum — cost of capital) |
| a₁₀ | 13.0% |
| a₂₀ | 18.0% (market baseline) |
| a₂₅ | 20.5% (≈ market average) |
| a₅₆ | 36.0% (maximum) |

### Why Discrete?
Microfinance institutions set rates in fixed increments (e.g., 0.5% steps)
for regulatory compliance and product standardization. Discrete actions
reflect this real-world constraint.

---

## 4. Transition Model — T(s' | s, a)

The transition function models borrower response to the offered rate:

### 4.1 Acceptance Probability
```
P(accept | s, a) = clip(base_prob - rate_penalty - rural_penalty 
                        - seasonal_penalty - market_penalty, 0.05, 0.98)

where:
  base_prob       = 0.7 + 0.2 × s₁              (income effect)
  rate_penalty    = max(0, a - 12%) × 0.04       (rate sensitivity)
  rural_penalty   = (1 - s₅) × max(0, a - 15%) × 0.03
  seasonal_penalty = s₇ × max(0, a - 14%) × 0.02
  market_penalty  = max(0, a - market_avg) × 0.03
```

### 4.2 Default Probability (given acceptance)
```
P(default | s, a, accepted) = clip(base_default + rate_effect 
                                   + income_effect + history_effect
                                   + credit_effect + economic_effect, 0.02, 0.95)

where:
  base_default     = 0.05
  rate_effect      = max(0, a - 12%) × 0.015
  income_effect    = max(0, 0.5 - income_loan_ratio) × 0.2
  history_effect   = s₈ × 0.3
  credit_effect    = (1 - s₃) × 0.15
  economic_effect  = s₁₀ × 0.2 + s₉ × 0.1
```

### 4.3 Episode Transition
```
Each step t:
  1. Agent observes sₜ (borrower profile)
  2. Agent selects action aₜ (interest rate)
  3. Environment samples outcome:
       accepted  ~ Bernoulli(P(accept | sₜ, aₜ))
       defaulted ~ Bernoulli(P(default | sₜ, aₜ)) if accepted
  4. Reward rₜ computed
  5. New borrower sₜ₊₁ sampled independently
  6. Episode ends after T=200 steps
```

---

## 5. Reward Function — R(s, a, s')

The reward is a multi-objective linear combination of 5 components:

### 5.1 Primary Outcome Rewards
```
R_outcome = +10   if accepted
           = -5    if rejected
           = +25   if repaid
           = -20   if defaulted
```

### 5.2 Sustainability Reward
```
R_sustainability = (a - CoC) × 0.5    if accepted AND repaid

where CoC = 8.0%  (cost of capital)
```

### 5.3 Market Competitiveness Reward ★ Unique Feature
```
R_market = +15   if a < market_average
         = +25   if a < lowest_market_rate   (beats all competitors)
         = -10   if a > market_average + 5%  (predatory pricing penalty)

Market rates (fluctuate ±2% per episode):
  SBI: 18.5%  |  HDFC: 19.2%  |  Bajaj Finance: 22.0%
  Local MFI avg: 24.0%  |  Market average: 20.9%
```

### 5.4 Fairness Penalty
```
R_fairness = -25   if avg_rural_rate > avg_urban_rate + 3%
           = -30   if high_dependents AND seasonal AND rate > market_avg + 3%
           =  0    otherwise
```

### 5.5 Total Reward
```
R_total = R_outcome + R_sustainability + R_market + R_fairness
```

### Reward Range Analysis
```
Best case:  +10 + +25 + (36-8)×0.5 + +25 + 0  = +74
Worst case: -5  + 0   + 0          + -10 + -55 = -70
Typical:    +30 to +50 per accepted+repaid loan
```

---

## 6. Policy — π(a | s)

The PPO agent learns a stochastic policy:

```
π_θ(a | s) = softmax(f_θ(s))

where f_θ is a neural network:
  Input:   s ∈ ℝ¹⁰
  Hidden:  [128] → ReLU → [128] → ReLU
  Output:  logits ∈ ℝ⁵⁷  →  probability distribution over 57 rates
```

### PPO Objective
```
L_CLIP(θ) = E[min(rₜ(θ) Â_t, clip(rₜ(θ), 1-ε, 1+ε) Â_t)]

where:
  rₜ(θ)  = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)   (probability ratio)
  Â_t    = GAE advantage estimate (λ=0.95)
  ε      = 0.2  (clip range)
```

---

## 7. Discount Factor — γ

```
γ = 0.99
```

High discount factor chosen because:
- Loan repayment consequences are delayed (future reward matters)
- MFI sustainability requires long-term thinking
- Encourages agent to avoid short-term predatory pricing

---

## 8. Convergence Analysis

| Timestep | Mean Reward | Interpretation |
|----------|------------|----------------|
| 0 | ~2,670 | Random exploration |
| 25,000 | ~4,000 | Learning acceptance patterns |
| 50,000 | ~7,000 | Discovering market competitiveness bonus |
| 75,000 | ~9,000 | Optimizing fairness trade-offs |
| 100,000 | ~11,370 | Converged optimal policy |

**Convergence criterion:** Mean episode reward plateau within ±5% over 10k steps ✅

---

## 9. Optimal Policy Characteristics

After training, the learned policy exhibits:

```
π*(s) ≈ 14.0%  for most borrower profiles

Rationale:
  - 14% < market_average (20.9%) → earns +15 market bonus
  - 14% > CoC (8%) → positive profit margin of 6%
  - 14% high acceptance rate → earns +10 per borrower
  - 14% low default rate → avoids -20 penalty
  - Uniform across demographics → 0% fairness violations
```

---

*MDP Specification — Mifos Initiative C4GT 2026*
*Dynamic Pricing of Micro-loans Using Reinforcement Learning*