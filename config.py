# Author:TK
# Date: 27-01-2026
# Desc: Main config file.

# Allowed plot colors
COLORS = {"blue", "red", "black", "green", "orange", "purple"}

# Default val shown when the page loads
DEFAULT_EXPRS = "x^2 - x + 2\nx^2 * math.sin(x) + 1"
DEFAULT_A = "10"
DEFAULT_B = "10"
DEFAULT_COLOR = "blue"

# Num of x-samples per plot
# Higher = smoother curve, but slower rendering
N_POINTS = 1000

# safety limiter which prevents user from plotting 100 expr at once
MAX_EXPRS = 6

