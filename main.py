import time
import logging
from core import SentimentClassifier, DNDDataset
from viz import compute_chunk_stats, plot_stats

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

    # Construct a single input prompt from the first chunk
    query = f"Classify sentiment: {dataset[0]}"
    logger.info(f"Classifying sentiment for first chunk...")

    # Run classification
    classification = model.classify(query)

    # Output result
    logger.info(f"Output: {classification}")
    logger.info(f"Time taken (s): {time.time() - start_time:.2f}")

    # Optionally compute and plot stats
    if enable_viz:
        logger.info("Computing and plotting text statistics...")
        chunks = [dataset[i] for i in range(len(dataset))]
        stats = compute_chunk_stats(chunks)
        plot_stats(stats)

if __name__ == "__main__":
    main()
