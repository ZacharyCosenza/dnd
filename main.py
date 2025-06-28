from core import SentimentClassifier, DNDDataset
import time
from viz import compute_chunk_stats, plot_stats

def main():

    viz = True
    time_start = time.time()

    path_model = 'models'
    path_data = 'data/notes.txt'

    # Get the pre-trained model
    model = SentimentClassifier(path_model)

    # Ge the dataset
    dataset = DNDDataset(path_data)

    query = "Classify sentiment: " + dataset[0]

    # Generate output
    c = model.classify(query)

    # Decode and print
    print("Output:", c)
    print("Time taken (s): ", time.time() - time_start)

    # Visualizations
    if viz:
        # Assuming `dataset` is your DNDDataset instance returning text chunks
        chunks = [dataset[i] for i in range(len(dataset))]
        stats = compute_chunk_stats(chunks)
        plot_stats(stats)


if __name__ == "__main__":
    main()
