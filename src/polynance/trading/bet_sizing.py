"""Bet sizing strategies for simulated trading."""

from typing import Optional, Literal


class SlowGrowthSizer:
    """Slow growth bet sizing strategy.

    Grows bet size slowly during win streaks, resets to base after loss.
    Combined with pause-after-loss, this strategy:
    - Avoids betting during dangerous post-loss periods (handled externally)
    - Grows bets conservatively during winning streaks
    - Caps growth to prevent oversized bets

    Based on analysis showing:
    - 94.7% win rate after wins vs 51.5% after losses
    - Best results: 10% growth per win, 2x cap, skip after loss
    - Calmar improvement: 64x vs anti-martingale
    """

    def __init__(
        self,
        base_bet: float = 25.0,
        growth_per_win: float = 0.10,  # 10% growth per consecutive win
        max_multiplier: float = 2.0,   # Cap at 2x base bet
        max_bet_pct: float = 0.05,     # 5% of bankroll cap
        min_bet_pct: float = 0.10,     # Floor at 10% of base bet
    ):
        """Initialize the slow growth sizer.

        Args:
            base_bet: Starting bet size (default: $25)
            growth_per_win: Percentage to grow per consecutive win (default: 0.10 = 10%)
            max_multiplier: Maximum multiplier on base bet (default: 2.0 = $50 max)
            max_bet_pct: Maximum bet as percentage of bankroll (default: 0.05 = 5%)
            min_bet_pct: Minimum bet as percentage of base bet (default: 0.10 = 10%)
        """
        self.base_bet = base_bet
        self.growth_per_win = growth_per_win
        self.max_multiplier = max_multiplier
        self.max_bet_pct = max_bet_pct
        self.min_bet_pct = min_bet_pct
        self.floor_bet = base_bet * min_bet_pct

    def calculate_bet_for_streak(self, win_streak: int, bankroll: float) -> float:
        """Calculate bet size based on current win streak.

        Args:
            win_streak: Number of consecutive wins (0 = first trade or after loss)
            bankroll: Current bankroll

        Returns:
            Bet size for this trade
        """
        # Calculate multiplier based on streak
        # After 0 wins: 1.0x, after 1 win: 1.1x, after 2 wins: 1.2x, etc.
        multiplier = 1.0 + (self.growth_per_win * win_streak)

        # Cap at max multiplier
        multiplier = min(multiplier, self.max_multiplier)

        # Calculate target bet
        target_bet = self.base_bet * multiplier

        # Apply bankroll cap
        max_allowed = bankroll * self.max_bet_pct
        actual_bet = min(target_bet, max_allowed)

        # Apply floor
        actual_bet = max(actual_bet, self.floor_bet)

        # Don't bet more than bankroll
        actual_bet = min(actual_bet, bankroll)

        return actual_bet

    def calculate_next_bet(
        self,
        actual_bet_used: float,
        bankroll: float,
        last_outcome: Optional[Literal["win", "loss"]] = None,
        win_streak: int = 0,
    ) -> float:
        """Calculate next bet size (compatibility method).

        For SlowGrowthSizer, the win_streak parameter drives sizing.
        This method exists for API compatibility with AntiMartingaleSizer.

        Args:
            actual_bet_used: The actual bet size used in the trade (ignored)
            bankroll: Current bankroll
            last_outcome: Result of last trade (used to adjust streak)
            win_streak: Current win streak count

        Returns:
            Next bet size
        """
        # If loss, reset to base
        if last_outcome == "loss":
            return self.calculate_bet_for_streak(0, bankroll)

        # If win, use provided streak (caller should track)
        return self.calculate_bet_for_streak(win_streak, bankroll)

    def reset_to_base(self) -> float:
        """Get the base bet size.

        Returns:
            Base bet size
        """
        return self.base_bet

    def get_config(self) -> dict:
        """Get the current configuration as a dictionary."""
        return {
            "type": "slow_growth",
            "base_bet": self.base_bet,
            "growth_per_win": self.growth_per_win,
            "max_multiplier": self.max_multiplier,
            "max_bet_pct": self.max_bet_pct,
            "min_bet_pct": self.min_bet_pct,
            "floor_bet": self.floor_bet,
        }

    def __repr__(self) -> str:
        return (
            f"SlowGrowthSizer("
            f"base=${self.base_bet}, "
            f"growth={self.growth_per_win*100}%/win, "
            f"cap={self.max_multiplier}x, "
            f"max={self.max_bet_pct*100}%)"
        )


class AntiMartingaleSizer:
    """Anti-martingale bet sizing strategy.

    Increases bet size after wins, decreases after losses.
    This strategy aims to maximize gains during winning streaks
    while minimizing losses during losing streaks.

    Configuration matches backtest_suite.py:
    - Base bet: $25
    - Win multiplier: 2x (double after win)
    - Loss multiplier: 0.5x (halve after loss)
    - Max bet: 5% of bankroll
    - Floor bet: 10% of base bet ($2.50)
    """

    def __init__(
        self,
        base_bet: float = 25.0,
        win_multiplier: float = 2.0,
        loss_multiplier: float = 0.5,
        max_bet_pct: float = 0.05,  # 5% of bankroll
        min_bet_pct: float = 0.10,  # 10% of base bet (floor)
    ):
        """Initialize the bet sizer.

        Args:
            base_bet: Starting bet size (default: $25)
            win_multiplier: Multiplier after a win (default: 2.0)
            loss_multiplier: Multiplier after a loss (default: 0.5)
            max_bet_pct: Maximum bet as percentage of bankroll (default: 0.05 = 5%)
            min_bet_pct: Minimum bet as percentage of base bet (default: 0.10 = 10%)
        """
        self.base_bet = base_bet
        self.win_multiplier = win_multiplier
        self.loss_multiplier = loss_multiplier
        self.max_bet_pct = max_bet_pct
        self.min_bet_pct = min_bet_pct
        self.floor_bet = base_bet * min_bet_pct  # $2.50 with defaults

    def calculate_next_bet(
        self,
        actual_bet_used: float,
        bankroll: float,
        last_outcome: Optional[Literal["win", "loss"]] = None,
    ) -> float:
        """Calculate next bet size based on last outcome.

        This matches backtest_suite.py behavior: multiply the ACTUAL bet used
        (after cap/floor applied), not the target bet size.

        Args:
            actual_bet_used: The actual bet size used in the trade (after cap/floor)
            bankroll: Current bankroll (after trade resolved)
            last_outcome: Result of last trade ('win', 'loss', or None for first trade)

        Returns:
            Next bet size, constrained by floor and cap
        """
        # Apply multiplier to ACTUAL bet used (matches backtest behavior)
        if last_outcome == "win":
            next_bet = actual_bet_used * self.win_multiplier
        elif last_outcome == "loss":
            next_bet = actual_bet_used * self.loss_multiplier
        else:
            # No outcome (first trade) - use actual bet
            next_bet = actual_bet_used

        # Apply floor (minimum bet)
        next_bet = max(next_bet, self.floor_bet)

        # Apply cap (maximum bet based on bankroll)
        max_allowed = bankroll * self.max_bet_pct
        next_bet = min(next_bet, max_allowed)

        # Ensure we don't bet more than we have
        next_bet = min(next_bet, bankroll)

        # Ensure at least floor bet (if bankroll allows)
        if bankroll >= self.floor_bet:
            next_bet = max(next_bet, self.floor_bet)
        else:
            next_bet = bankroll  # Bet remaining bankroll

        return next_bet

    def calculate_actual_bet(
        self,
        current_bet: float,
        bankroll: float,
    ) -> float:
        """Calculate actual bet size for the current trade.

        This applies the cap and floor without modifying the bet size.
        Use this when entering a trade.

        Args:
            current_bet: Target bet size
            bankroll: Current bankroll

        Returns:
            Actual bet size to use for this trade
        """
        # Apply cap
        actual_bet = min(current_bet, bankroll * self.max_bet_pct)

        # Apply floor
        actual_bet = max(actual_bet, self.floor_bet)

        # Don't bet more than bankroll
        actual_bet = min(actual_bet, bankroll)

        return actual_bet

    def reset_to_base(self) -> float:
        """Get the base bet size for resetting after a streak breaks.

        Returns:
            Base bet size
        """
        return self.base_bet

    def get_config(self) -> dict:
        """Get the current configuration as a dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "base_bet": self.base_bet,
            "win_multiplier": self.win_multiplier,
            "loss_multiplier": self.loss_multiplier,
            "max_bet_pct": self.max_bet_pct,
            "min_bet_pct": self.min_bet_pct,
            "floor_bet": self.floor_bet,
        }

    def __repr__(self) -> str:
        return (
            f"AntiMartingaleSizer("
            f"base={self.base_bet}, "
            f"win_mult={self.win_multiplier}x, "
            f"loss_mult={self.loss_multiplier}x, "
            f"max={self.max_bet_pct*100}%, "
            f"floor=${self.floor_bet:.2f})"
        )
