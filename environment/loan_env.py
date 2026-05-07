"""
loan_env.py - Custom Gymnasium Environment for Micro-loan Dynamic Pricing
=========================================================================
Mifos Initiative - DMP 2026: Dynamic Pricing of Micro-loans Using RL

This environment simulates a microfinance institution setting interest rates
for borrowers. The RL agent must balance profit, fairness, and sustainability
while remaining competitive with market rates.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any


class MicroLoanEnv(gym.Env):
    """
    Custom Gymnasium environment for micro-loan interest rate optimization.

    The agent observes borrower profiles and economic conditions, then
    selects an interest rate. Rewards are multi-objective: balancing
    profitability, loan acceptance, repayment, fairness, and market
    competitiveness.

    Observation Space (10 features, all normalized to [0, 1]):
        - monthly_income: borrower's monthly income
        - loan_amount_requested: loan size relative to max
        - credit_score: creditworthiness (0=poor, 1=excellent)
        - employment_type: 0=farmer, 1=trader, 2=salaried (normalized)
        - location: 0=rural, 1=semi-urban, 2=urban (normalized)
        - dependents: number of dependents (0-6, normalized)
        - seasonal_income: boolean flag for seasonal earners
        - previous_defaults: number of past defaults (0-3, normalized)
        - inflation_rate: current economic inflation
        - unemployment_rate: current economic unemployment

    Action Space:
        Discrete(57): interest rates from 8% to 36% in 0.5% steps
    """

    metadata = {"render_modes": ["human"]}

    # Cost of capital - the MFI's own borrowing cost
    COST_OF_CAPITAL: float = 8.0

    # Market competitor rates (simulated Indian MFI landscape)
    BASE_MARKET_RATES: Dict[str, float] = {
        "SBI": 18.5,
        "HDFC": 19.2,
        "Bajaj_Finance": 22.0,
        "Local_MFI_avg": 24.0,
        "market_average": 20.9,
    }

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the MicroLoan environment.

        Args:
            render_mode: Optional rendering mode ('human' for console output)
        """
        super().__init__()
        self.render_mode = render_mode

        # ── Action Space ────────────────────────────────────────────────────
        # 57 discrete actions: 8% + (action * 0.5%) → range [8%, 36%]
        self.action_space = spaces.Discrete(57)
        self.min_rate: float = 8.0
        self.rate_step: float = 0.5

        # ── Observation Space ────────────────────────────────────────────────
        # 10 features, all normalized to [0.0, 1.0]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        # ── Market Rates (fluctuate each episode) ───────────────────────────
        self.market_rates: Dict[str, float] = dict(self.BASE_MARKET_RATES)
        self.market_average: float = self.BASE_MARKET_RATES["market_average"]
        self.lowest_market_rate: float = min(self.BASE_MARKET_RATES.values())

        # ── Fairness Tracking ────────────────────────────────────────────────
        # Track rates given to rural vs urban borrowers with similar financials
        self.demographic_rate_history: Dict[str, list] = {
            "rural": [],
            "semi_urban": [],
            "urban": [],
        }

        # ── Episode Tracking ─────────────────────────────────────────────────
        self.current_step: int = 0
        self.max_steps: int = 200  # steps per episode
        self.current_borrower: Optional[np.ndarray] = None

        # Store last borrower info for render
        self._last_info: Dict[str, Any] = {}

    # ────────────────────────────────────────────────────────────────────────
    # Environment Core Methods
    # ────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Generates new market rate fluctuations and a fresh borrower profile.

        Returns:
            observation: Initial borrower profile observation
            info: Additional episode info dict
        """
        super().reset(seed=seed)

        # Fluctuate market rates slightly each episode (±2%)
        self.market_rates = {
            k: v + self.np_random.uniform(-2.0, 2.0)
            for k, v in self.BASE_MARKET_RATES.items()
        }
        self.market_average = self.market_rates["market_average"]
        self.lowest_market_rate = min(self.market_rates.values())

        # Clear fairness tracking for new episode
        self.demographic_rate_history = {"rural": [], "semi_urban": [], "urban": []}

        self.current_step = 0
        self.current_borrower = self._generate_borrower()

        info = {"market_rates": self.market_rates}
        return self.current_borrower.copy(), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one pricing decision.

        Args:
            action: Integer in [0, 56] representing interest rate choice

        Returns:
            observation: Next borrower profile
            reward: Multi-objective reward signal
            terminated: True if episode ended naturally
            truncated: True if max_steps reached
            info: Diagnostic information
        """
        # Convert discrete action → actual interest rate
        interest_rate: float = self.min_rate + action * self.rate_step

        # Simulate loan outcome based on borrower profile + rate
        accepted: bool = self._simulate_acceptance(interest_rate)
        repaid: bool = False
        defaulted: bool = False

        if accepted:
            defaulted = self._simulate_default(interest_rate)
            repaid = not defaulted

        # Compute multi-objective reward
        reward, reward_breakdown = self._compute_reward(
            interest_rate, accepted, repaid, defaulted
        )

        # Update fairness history
        self._update_fairness_history(interest_rate)

        # Advance to next borrower
        self.current_step += 1
        self.current_borrower = self._generate_borrower()

        terminated: bool = False
        truncated: bool = self.current_step >= self.max_steps

        info = {
            "interest_rate": interest_rate,
            "accepted": accepted,
            "repaid": repaid,
            "defaulted": defaulted,
            "market_average": self.market_average,
            "reward_breakdown": reward_breakdown,
        }
        self._last_info = info

        if self.render_mode == "human":
            self.render()

        return self.current_borrower.copy(), reward, terminated, truncated, info

    # ────────────────────────────────────────────────────────────────────────
    # Borrower Generation
    # ────────────────────────────────────────────────────────────────────────

    def _generate_borrower(self) -> np.ndarray:
        """
        Generate a synthetic borrower profile.

        Profiles reflect the diversity of microfinance clients in India:
        farmers, traders, salaried workers across rural/urban areas.

        Returns:
            np.ndarray: 10-feature observation vector (all in [0, 1])
        """
        rng = self.np_random

        # Monthly income: ₹3,000 - ₹50,000 (normalized)
        monthly_income_raw = rng.uniform(3000, 50000)
        monthly_income = (monthly_income_raw - 3000) / (50000 - 3000)

        # Loan amount: ₹5,000 - ₹500,000 (normalized)
        loan_amount = rng.uniform(0.0, 1.0)

        # Credit score: 300-850 (normalized)
        credit_score = rng.beta(2, 2)  # Beta distribution → realistic spread

        # Employment type: 0=farmer, 1=trader, 2=salaried
        employment_type = rng.integers(0, 3) / 2.0

        # Location: 0=rural, 1=semi-urban, 2=urban
        location = rng.integers(0, 3) / 2.0

        # Dependents: 0-6 (normalized)
        dependents = rng.integers(0, 7) / 6.0

        # Seasonal income flag (common for farmers)
        seasonal_income = float(rng.random() < 0.35)  # 35% chance

        # Previous defaults: 0-3 (normalized)
        # Skewed toward 0 (most borrowers have clean records)
        prev_defaults = rng.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.1, 0.05]) / 3.0

        # Economic conditions (macro-level, same within an episode)
        inflation = rng.uniform(0.02, 0.15)   # 2% to 15%
        unemployment = rng.uniform(0.03, 0.25)  # 3% to 25%

        # Store raw values for reward computation
        self._raw_income = monthly_income_raw
        self._raw_location = location
        self._raw_credit = credit_score
        self._raw_loan = loan_amount * 500000 + 5000
        self._raw_employment = employment_type
        self._raw_defaults = prev_defaults

        return np.array([
            monthly_income,
            loan_amount,
            credit_score,
            employment_type,
            location,
            dependents,
            seasonal_income,
            prev_defaults,
            inflation,
            unemployment,
        ], dtype=np.float32)

    # ────────────────────────────────────────────────────────────────────────
    # Loan Outcome Simulation
    # ────────────────────────────────────────────────────────────────────────

    def _simulate_acceptance(self, rate: float) -> bool:
        """
        Simulate whether the borrower accepts the loan at the given rate.

        Acceptance probability decreases with higher rates.
        Rural and seasonal borrowers are more rate-sensitive.

        Args:
            rate: Interest rate offered (%)

        Returns:
            bool: True if borrower accepts the loan
        """
        obs = self.current_borrower
        monthly_income_norm = obs[0]
        location_norm = obs[4]         # 0=rural, 0.5=semi-urban, 1=urban
        seasonal = obs[6]

        # Base acceptance: higher income → more willing to borrow
        base_prob = 0.7 + 0.2 * monthly_income_norm

        # Rate sensitivity: every 1% above 12% reduces acceptance
        rate_penalty = max(0, (rate - 12.0)) * 0.04

        # Rural borrowers are more sensitive to high rates
        rural_penalty = (1.0 - location_norm) * max(0, (rate - 15.0)) * 0.03

        # Seasonal income borrowers are more risk-averse
        seasonal_penalty = seasonal * max(0, (rate - 14.0)) * 0.02

        # Market comparison: if rate > market average, borrowers look elsewhere
        market_penalty = max(0, (rate - self.market_average)) * 0.03

        acceptance_prob = np.clip(
            base_prob - rate_penalty - rural_penalty - seasonal_penalty - market_penalty,
            0.05, 0.98
        )

        return bool(self.np_random.random() < acceptance_prob)

    def _simulate_default(self, rate: float) -> bool:
        """
        Simulate whether the borrower defaults on the loan.

        Higher rates, lower income, and previous defaults all increase
        the probability of default. Economic conditions also matter.

        Args:
            rate: Interest rate charged (%)

        Returns:
            bool: True if borrower defaults (fails to repay)
        """
        obs = self.current_borrower
        monthly_income_norm = obs[0]
        loan_amount_norm = obs[1]
        credit_score = obs[2]
        prev_defaults_norm = obs[7]
        inflation = obs[8]
        unemployment = obs[9]

        # Income-to-loan ratio: low income + high loan = higher default risk
        income_loan_ratio = monthly_income_norm / (loan_amount_norm + 0.01)

        # Base default probability
        base_default = 0.05

        # Higher rate → harder to repay
        rate_effect = max(0, (rate - 12.0)) * 0.015

        # Poor income-to-loan ratio increases risk
        income_effect = max(0, 0.5 - income_loan_ratio) * 0.2

        # Previous defaults are strong predictors
        default_history_effect = prev_defaults_norm * 0.3

        # Poor credit score increases risk
        credit_effect = (1.0 - credit_score) * 0.15

        # Economic conditions: high unemployment → more defaults
        economic_effect = unemployment * 0.2 + inflation * 0.1

        default_prob = np.clip(
            base_default + rate_effect + income_effect
            + default_history_effect + credit_effect + economic_effect,
            0.02, 0.95
        )

        return bool(self.np_random.random() < default_prob)

    # ────────────────────────────────────────────────────────────────────────
    # Reward Function
    # ────────────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        rate: float,
        accepted: bool,
        repaid: bool,
        defaulted: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Multi-objective reward function balancing profit, fairness,
        and market competitiveness.

        Args:
            rate: Interest rate charged
            accepted: Whether borrower accepted the loan
            repaid: Whether borrower fully repaid
            defaulted: Whether borrower defaulted

        Returns:
            total_reward: Scalar reward for the RL agent
            breakdown: Dict of individual reward components for logging
        """
        reward = 0.0
        breakdown: Dict[str, float] = {}

        # ── Primary Outcome Rewards ──────────────────────────────────────────
        if accepted:
            breakdown["acceptance"] = 10.0
            reward += 10.0
        else:
            breakdown["rejection"] = -5.0
            reward -= 5.0

        if repaid:
            breakdown["repayment"] = 25.0
            reward += 25.0

        if defaulted:
            breakdown["default"] = -20.0
            reward -= 20.0

        # ── Sustainability / Profit Margin ───────────────────────────────────
        # Only earns profit if loan is accepted and repaid
        if accepted and repaid:
            profit_margin = rate - self.COST_OF_CAPITAL
            sustainability = profit_margin * 0.5
            breakdown["sustainability"] = sustainability
            reward += sustainability

        # ── Market Competitiveness Bonus (UNIQUE FEATURE) ────────────────────
        if rate < self.market_average:
            market_bonus = 15.0
            breakdown["below_market"] = market_bonus
            reward += market_bonus

            # Extra bonus for beating the lowest competitor rate
            if rate < self.lowest_market_rate:
                lowest_bonus = 25.0
                breakdown["below_lowest_market"] = lowest_bonus
                reward += lowest_bonus

        elif rate > self.market_average + 5.0:
            # Penalty for pricing far above market (predatory pricing)
            market_penalty = -10.0
            breakdown["above_market"] = market_penalty
            reward += market_penalty

        # ── Fairness Penalty ─────────────────────────────────────────────────
        fairness_penalty = self._compute_fairness_penalty(rate)
        if fairness_penalty != 0:
            breakdown["fairness"] = fairness_penalty
            reward += fairness_penalty

        breakdown["total"] = reward
        return reward, breakdown

    def _compute_fairness_penalty(self, rate: float) -> float:
        """
        Compute fairness violation penalties.

        Checks if rural borrowers are being systematically charged more
        than urban borrowers with similar financial profiles.

        Args:
            rate: Rate being considered for current borrower

        Returns:
            float: Penalty (negative) if fairness violated, else 0
        """
        penalty = 0.0
        obs = self.current_borrower
        location_norm = obs[4]   # 0=rural, 0.5=semi_urban, 1=urban

        # Determine demographic group
        if location_norm == 0.0:
            location_group = "rural"
        elif location_norm == 0.5:
            location_group = "semi_urban"
        else:
            location_group = "urban"

        # Need at least some history to compare
        if len(self.demographic_rate_history["rural"]) >= 5 and \
           len(self.demographic_rate_history["urban"]) >= 5:

            avg_rural_rate = np.mean(self.demographic_rate_history["rural"])
            avg_urban_rate = np.mean(self.demographic_rate_history["urban"])

            # Rural borrowers should not be charged >3% more than urban
            if avg_rural_rate > avg_urban_rate + 3.0:
                penalty -= 25.0  # Fairness violation

        # Fairness across demographic groups logged for analysis
        # (Gender proxy via dependents + seasonal patterns)
        # High dependents + seasonal income → proxy for female-headed households
        dependents_norm = obs[5]
        seasonal = obs[6]
        if dependents_norm > 0.5 and seasonal == 1.0:
            if rate > self.market_average + 3.0:
                penalty -= 30.0  # Penalty for apparent gender proxy penalization

        return penalty

    def _update_fairness_history(self, rate: float) -> None:
        """
        Record rate offered per demographic group for fairness tracking.

        Args:
            rate: Interest rate charged to current borrower
        """
        obs = self.current_borrower
        location_norm = obs[4]

        if location_norm == 0.0:
            self.demographic_rate_history["rural"].append(rate)
        elif location_norm == 0.5:
            self.demographic_rate_history["semi_urban"].append(rate)
        else:
            self.demographic_rate_history["urban"].append(rate)

        # Keep history bounded
        for key in self.demographic_rate_history:
            if len(self.demographic_rate_history[key]) > 500:
                self.demographic_rate_history[key] = \
                    self.demographic_rate_history[key][-500:]

    # ────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ────────────────────────────────────────────────────────────────────────

    def action_to_rate(self, action: int) -> float:
        """Convert discrete action index to interest rate percentage."""
        return self.min_rate + action * self.rate_step

    def render(self) -> None:
        """Human-readable rendering of the last step."""
        if self.render_mode == "human" and self._last_info:
            info = self._last_info
            obs = self.current_borrower
            print(
                f"Step {self.current_step:4d} | "
                f"Rate: {info['interest_rate']:5.1f}% | "
                f"Market avg: {self.market_average:5.1f}% | "
                f"Accepted: {info['accepted']} | "
                f"Repaid: {info['repaid']} | "
                f"Reward: {info['reward_breakdown'].get('total', 0):+.1f}"
            )

    def close(self) -> None:
        """Clean up environment resources."""
        pass