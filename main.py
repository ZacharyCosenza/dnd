import time
import logging
from core import SentimentClassifier, DNDDataset
from viz import compute_chunk_stats, plot_stats, plot_sentiment, plot_heatmap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Entry point for running sentiment classification and visualization on a dataset.
    """
    # Toggle to enable/disable visualizations
    enable_viz = True

    # Start the timer
    start_time = time.time()

    # Paths
    path_model = 'models'  # Path to pre-trained model
    path_data = ['data/old_notes.txt', 'data/notes.txt']  # Text file(s) to process

    # Load sentiment classifier
    logger.info(f"Loading sentiment classifier from: {path_model}")
    model = SentimentClassifier(path_model)

    # Load dataset and tokenize into chunks
    logger.info(f"Loading dataset from: {path_data}")
    dataset = DNDDataset(path_data)

    few_shots = {'positive': "Goratur you did a great job!",
                 'negative': "Goratur was such a pedo.",
                 'neural': "John you did a great job!"}

    semantics = {'Overall Sentiment': "Classify sentiment [positive, negative, neutral]: ",
                 'Goratur Sentiment (Zero-Shot)': "Classify sentiment about Goratur [positive, negative, neutral]: ",
                 'Gortaur Sentiment (Few-Shot)': f"Classify sentiment about Goratur using these few-shot examples \n {[v + ': sentiment is ' + k for k, v in few_shots.items()]} [positive, negative, neutral]: "}

    results = {}
    for name, semantic in semantics.items():
        results[name] = []
        for i in range(0, len(dataset), 5):
            chunk = dataset[i]
            query = semantic + chunk
            c = model.classify(query)
            results[name].append(c)

    # Decode and print
    print("Time taken (s): ", time.time() - start_time)

    # Optionally compute and plot stats
    if enable_viz:
        logger.info("Computing and plotting text statistics...")
        chunks = [dataset[i] for i in range(len(dataset))]
        stats = compute_chunk_stats(chunks)
        plot_stats(stats)
        plot_heatmap(chunks)
        plot_sentiment(results)

if __name__ == "__main__":
    main()
