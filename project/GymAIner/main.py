import os

from structures import ConvLTSMModel

DATASET_PATH = "dataset\\Fit3D Video Dataset"

def main():
    settings = {
        "dataset": {
            "path": DATASET_PATH, 
            "sequence_length": 20, 
            "resize_width": 64, 
            "resize_height": 64, 
            "color_channels": 3, 
            "validation_ratio": 0.2, 
            "test_ratio": 0.2,
        }, 
        
        "model": {
            "filter_counts": [4, 8, 14, 16],
            "kernel_sizes": [(3, 3), (3, 3), (3, 3), (3, 3)],
            "activations": ["tanh", "tanh", "tanh", "tanh", "softmax"],
            "data_formats": ["channels_last", "channels_last", "channels_last", "channels_last", "channels_last", "channels_last", "channels_last", "channels_last"],
            "recurrent_dropouts": [0.2, 0.2, 0.2, 0.2],
            "return_sequences": [True, True, True, True],
            "pool_sizes": [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
            "paddings": ["same", "same", "same", "same"],
            "dropouts": [0.2, 0.2, 0.2],
        }, 
        
        "early_stopping": {
            "count": 1, 
            "monitors": ["val_loss"], 
            "patiences": [10], 
            "modes": ["min"], 
            "restore_best_weights": [True]
        }, 
        
        "compiling": {
            "loss": "categorical_crossentropy", 
            "optimizer": "Adam", 
            "metrics": ["accuracy"]
        }, 
        
        "training": {
            "epochs": 50, 
            "batch_size": 4, 
            "shuffle": True
        }, 
        
        "statistics": {
            "graph_size": (12, 6)
        }
    }
    
    model = ConvLTSMModel(settings)
    model.execute()
    
    return 0


if __name__ == "__main__":
    main()
    