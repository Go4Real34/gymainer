import os

from structures import DatasetHandler

DATASET_PATH = os.path.join("dataset", "Fit3D Video Dataset")

def main():
    dataset_handler = DatasetHandler(DATASET_PATH, 75, 128, 128, 3, 0.2, 0.2)
    dataset_handler.init()
    
    return 0


if __name__ == "__main__":
    main()
    