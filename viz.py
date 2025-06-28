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
<<<<<<< HEAD
    logger.info(f"Saved plot as {filename}")
=======
    print(f"Saved plot as {filename}")

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_sentiment(results):
        
    def running_average(arr, w=5):
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    plt.figure(figsize=(13, 5))

    colors = {'Overall Sentiment': 'blue',
              'Goratur Sentiment (Zero-Shot)': 'red',
              'Gortaur Sentiment (Few-Shot)': 'green'}

    for name, result in results.items():

        # Map sentiment to numeric values
        mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
        values = np.array([mapping[s] for s in result])

        avg = running_average(values)
        cumavg = np.cumsum(values) / (np.arange(len(values)) + 1)

        plt.plot(range(len(avg)), avg, linewidth=2, color = colors[name], label=f'Running Average {name}', alpha = 0.5, linestyle = '--')
        plt.plot(range(len(cumavg)), cumavg, linewidth=2, color = colors[name] , label=f'Cumulative Average {name}')

    plt.xlabel('Time')
    plt.ylabel('Sentiment Value')
    plt.title('Semantic Scores over Time')
    plt.legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_semantics.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot as {filename}")
<<<<<<< HEAD
>>>>>>> 35c05f2 (added sentiment tracker)
=======

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def compute_cooccurrence_matrix(chunks, words):
    word_set = set(words)
    word_to_idx = {word: i for i, word in enumerate(words)}
    matrix = np.zeros((len(words), len(words)), dtype=int)

    for chunk in chunks:
        tokens = set(chunk.lower().split())  # simple tokenization
        present = [w for w in words if w in tokens]
        for i in range(len(present)):
            for j in range(i, len(present)):
                idx_i, idx_j = word_to_idx[present[i]], word_to_idx[present[j]]
                matrix[idx_i][idx_j] += 1
                if idx_i != idx_j:
                    matrix[idx_j][idx_i] += 1  # symmetric

    return matrix

def plot_heatmap(chunks, title="Word Co-occurrence Heatmap"):
    words = ['goratur', 'broll', 'blampington', 'fenris']
    matrix = compute_cooccurrence_matrix(chunks, words)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=words, yticklabels=words, cmap="Blues", annot=True, fmt="d")
    plt.title(title)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_heatmap.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot as {filename}")
>>>>>>> 0257cd2 (updated viz)
