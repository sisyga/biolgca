import numpy as np

def estimate_entropy(data):
    # Compute histogram of data using numpy's automatic bin selection
    hist, bin_edges = np.histogram(data, bins='auto', density=True)

    # Calculate bin widths
    bin_width = bin_edges[1] - bin_edges[0]
    print(hist.sum() * bin_width)

    # Create a masked array to handle zero probabilities
    probabilities = np.ma.masked_equal(hist, 0)

    # Compute the entropy
    entropy = -np.ma.sum(probabilities * np.ma.log2(probabilities) * bin_width)

    return entropy

# Generate a dataset of 1000 random float values from a normal distribution
mean = 0
std_dev = 1
data = np.random.normal(mean, std_dev, 10000)

# Calculate the estimated entropy
estimated_entropy = estimate_entropy(data)
print("Estimated entropy:", estimated_entropy)

# Calculate the analytical entropy
variance = std_dev**2
analytical_entropy = 0.5 * np.log2(2 * np.pi * np.e * variance)
print("Analytical entropy:", analytical_entropy)