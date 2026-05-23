ROLLING_WINDOW = 7
STRONG_BUY = 2
STRONG_SELL = -2
HOLD = 0
SELL = -1
BUY = 1
INITIAL_PORTFOLIO = 1000.0
WINDOW_SIZE = 256
SPLIT_RATIO = 0.7685
THRESHOLD = 0.03
OPTIM_RATIO = 0.75
WINDOW_SIZE_DATA = 256
TRAIN_OPTIM = 64

# Updated PPO parameters for better convergence
FC1_DIMS = 128  # Reduced for stability
FC2_DIMS = 64   # Reduced for stability
GAMMA = 0.95    # Reduced discount for more immediate rewards
ALPHA = 3e-4    # Adjusted learning rate
GAE_LAMBDA = 0.90  # Reduced for more stable advantages
POLICY_CLIP = 0.2  # Tighter clipping
BATCH_SIZE = 512   # More balanced batch size
N = 1024        # Reduced rollout length
N_EPOCHS = 10   # More epochs for better learning