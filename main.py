import time
from core import SentimentClassifier, DNDDataset

def main():

    start_time = time.time()

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
    print("Elapsed time: {:.2f}s".format(time.time() - start_time))

if __name__ == "__main__":
    main()
