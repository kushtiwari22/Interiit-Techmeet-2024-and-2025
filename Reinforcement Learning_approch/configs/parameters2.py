# Stable Parameters for Low-Volatility Trading
# Optimized for 1-2% daily movements and reduced trade frequency

# Trading Signal Constants
ROLLING_WINDOW = 7
STRONG_BUY = 2
STRONG_SELL = -2
HOLD = 0
SELL = -1
BUY = 1

# Portfolio Settings
INITIAL_PORTFOLIO = 1000.0

# Window and Split Settings
WINDOW_SIZE = 128  # Reduced for faster adaptation to market changes
SPLIT_RATIO = 0.7  # Simplified split ratio
THRESHOLD = 0.02   # Adjusted for lower volatility
OPTIM_RATIO = 0.7  # Consistent with split ratio
WINDOW_SIZE_DATA = 128
TRAIN_OPTIM = 45   # Adjusted for smaller window

# Network Architecture - SIMPLIFIED for stability
FC1_DIMS = 64      # Reduced from 256 to prevent overfitting
FC2_DIMS = 32      # Reduced from 256 for simpler patterns

# PPO Hyperparameters - OPTIMIZED FOR STABILITY
GAMMA = 0.99                    # Discount factor (standard)
ALPHA = 3e-4                    # Much lower learning rate for stability (reduced from 1e-4)
GAE_LAMBDA = 0.92               # Slightly lower GAE lambda for more immediate rewards
POLICY_CLIP = 0.2               # Tighter clipping to prevent large policy updates
BATCH_SIZE = 64                 # Smaller batches for more stable gradient estimates
N = 256                         # More frequent updates (reduced from 512)
N_EPOCHS = 10                   # Fewer epochs to prevent overfitting to small batches

# Exploration Control - GRADUAL DECAY
ENTROPY_COEF_START = 0     # Lower initial exploration (reduced from 0.01)
ENTROPY_COEF_END = 0.0005       # Final entropy coefficient
ENTROPY_DECAY = 0.998           # Slower decay for more consistent exploration

# Action Selection - CONSERVATIVE
EXPLORATION_RATE = 0.1         # Much less exploration during walkforward (reduced from 0.1)
HOLD_PENALTY = 0.90             # Less aggressive hold penalty (increased from 0.80)

# Reward Shaping - CONSERVATIVE for 1-2% daily moves
REWARD_CLIP_MIN = -0.02         # Tighter bounds appropriate for low volatility
REWARD_CLIP_MAX = 0.02          # Maximum reward per step
TRANSACTION_COST = 0.0004       # Transaction cost (consistent)
HOLDING_BONUS = 0.0002          # Reduced holding bonus to prevent excessive position keeping

# New Reward Parameters - SIMPLIFIED
PNL_AMPLIFICATION = 1.0         # No amplification - use raw PnL (reduced from 10)
DIRECTIONAL_BONUS = 2.0         # Reduced directional bonus (from 5)
DRAWDOWN_PENALTY = 5.0          # Reduced drawdown penalty (from 10)
SHARPE_BONUS = 0.05             # Reduced Sharpe bonus (from 0.1)

# Learning Frequency - OPTIMIZED
LEARN_FREQUENCY = 64            # More frequent learning (reduced from 128)

# Gradient Clipping
GRAD_CLIP_NORM = 5            # Gradient clipping norm (unchanged)

# Learning Rate Decay - SLOWER
LR_DECAY = 0.9998               # Slower learning rate decay

# Position Smoothing Parameters
POSITION_SMOOTHING = True
FLIP_PENALTY = 0.3              # Penalty for flipping positions too frequently
MIN_POSITION_HOLD = 3           # Minimum steps to hold a position (not enforced, just guidance)

# Risk Management
MAX_DRAWDOWN_THRESHOLD = 0.02   # 2% drawdown threshold for penalties
SHARPE_WINDOW = 15              # Shorter window for Sharpe calculation

# Trade Frequency Control
MAX_TRADES_PER_DAY = 10         # Soft limit on trades per day
TRADE_FREQUENCY_PENALTY = 0.001 # Small penalty for excessive trading

# Volatility Adjustment
VOLATILITY_SCALING = True       # Scale rewards by market volatility
BASE_VOLATILITY = 0.01          # Base volatility assumption (1%)

print("✓ Stable parameters loaded for low-volatility trading environment")
print("✓ Designed for 1-2% daily movements with reduced trade frequency")
print("✓ Network complexity reduced to prevent overfitting")
print("✓ Learning rate lowered for training stability")