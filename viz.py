import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from datetime import datetime

def compute_chunk_stats(chunks):
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

    return stats

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def moving_average(x, w=3):
    if len(x) < w:
        return x  # no smoothing if not enough points
    return np.convolve(x, np.ones(w)/w, mode='same')

def plot_stats(stats, window=15):
    # Compute smoothed stats
    smoothed_avg_len = moving_average(stats["avg_word_length"], w=window)
    smoothed_unique_count = moving_average(stats["unique_word_count"], w=window)
    smoothed_common_freq = moving_average(stats["most_common_word_freq"], w=window)

    # Trim edges: remove floor(window/2) elements from start and end
    trim = window // 2
    x = np.arange(len(stats["avg_word_length"]))
    x_trimmed = x[trim:-trim]
    avg_len_trimmed = smoothed_avg_len[trim:-trim]
    unique_count_trimmed = smoothed_unique_count[trim:-trim]
    common_freq_trimmed = smoothed_common_freq[trim:-trim]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(x_trimmed, avg_len_trimmed, marker='o')
    plt.title("Average Word Length (smoothed)")
    plt.xlabel("Chunk")
    plt.ylabel("Avg Word Length")
    plt.xlim(x_trimmed[0], x_trimmed[-1])

    plt.subplot(1, 3, 2)
    plt.plot(x_trimmed, unique_count_trimmed, marker='o', color='orange')
    plt.title("Unique Word Count (smoothed)")
    plt.xlabel("Chunk")
    plt.ylabel("Unique Words")
    plt.xlim(x_trimmed[0], x_trimmed[-1])

    plt.subplot(1, 3, 3)
    plt.plot(x_trimmed, common_freq_trimmed, marker='o', color='red')
    plt.title("Most Common Word Frequency (smoothed)")
    plt.xlabel("Chunk")
    plt.ylabel("Frequency")
    plt.xlim(x_trimmed[0], x_trimmed[-1])

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_metrics.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot as {filename}")
