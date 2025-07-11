import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_chunk_stats(chunks: List[str]) -> Dict[str, List[float]]:
    """
    Compute basic statistics for each chunk of text.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        Dict[str, List[float]]: A dictionary with keys:
            - avg_word_length: average word length in the chunk.
            - unique_word_count: number of unique words in the chunk.
            - most_common_word_freq: frequency of the most common word.
    """
    stats = {
        "avg_word_length": [],
        "unique_word_count": [],
        "most_common_word_freq": []
    }

    for chunk in chunks:
        words = chunk.split()
        unique_words = set(words)
        word_counts = Counter(words)

        stats["avg_word_length"].append(np.mean([len(w) for w in words]) if words else 0)
        stats["unique_word_count"].append(len(unique_words))
        stats["most_common_word_freq"].append(word_counts.most_common(1)[0][1] if words else 0)

    logger.info(f"Computed statistics for {len(chunks)} chunks.")
    return stats

def moving_average(x: List[float], w: int = 3) -> np.ndarray:
    """
    Compute moving average with window size w.

    Args:
        x (List[float]): Sequence of numeric values.
        w (int): Window size.

    Returns:
        np.ndarray: Smoothed sequence.
    """
    if len(x) < w:
        logger.warning("Not enough data for smoothing, returning raw values.")
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode='same')

def plot_stats(stats: Dict[str, List[float]], window: int = 15) -> None:
    """
    Plot smoothed text statistics over time and save as a PNG.

    Args:
        stats (Dict[str, List[float]]): Output from compute_chunk_stats.
        window (int): Smoothing window size.
    """
    logger.info("Generating visualizations...")

    # Smooth the metrics
    smoothed_avg_len = moving_average(stats["avg_word_length"], w=window)
    smoothed_unique_count = moving_average(stats["unique_word_count"], w=window)
    smoothed_common_freq = moving_average(stats["most_common_word_freq"], w=window)

    # Trim window padding to avoid edge artifacts
    trim = window // 2
    x = np.arange(len(stats["avg_word_length"]))
    if len(x) <= window:
        logger.warning("Too few chunks for plotting after trimming.")
        return

    x_trimmed = x[trim:-trim]
    avg_len_trimmed = smoothed_avg_len[trim:-trim]
    unique_count_trimmed = smoothed_unique_count[trim:-trim]
    common_freq_trimmed = smoothed_common_freq[trim:-trim]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(x_trimmed, avg_len_trimmed, marker='o')
    plt.title("Average Word Length (Smoothed)")
    plt.xlabel("Chunk Index")
    plt.ylabel("Avg Word Length")
    plt.xlim(x_trimmed[0], x_trimmed[-1])

    plt.subplot(1, 3, 2)
    plt.plot(x_trimmed, unique_count_trimmed, marker='o', color='orange')
    plt.title("Unique Word Count (Smoothed)")
    plt.xlabel("Chunk Index")
    plt.ylabel("Unique Words")
    plt.xlim(x_trimmed[0], x_trimmed[-1])

    plt.subplot(1, 3, 3)
    plt.plot(x_trimmed, common_freq_trimmed, marker='o', color='red')
    plt.title("Most Common Word Frequency (Smoothed)")
    plt.xlabel("Chunk Index")
    plt.ylabel("Frequency")
    plt.xlim(x_trimmed[0], x_trimmed[-1])

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_metrics.png"
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved plot as {filename}")
