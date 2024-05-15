import os

from structures import ConvLTSMModel, LRCNModel

DATASET_PATH = os.path.join("dataset", "Fit3D Video Dataset")

def conv_lstm_model_execute():
    conv_ltsm_model_settings = {
        "dataset": {
            "path": DATASET_PATH, 
            "sequence_length": 150, 
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
            "learning_rate": 0.001, 
            "metrics": ["accuracy"]
        }, 
        
        "training": {
            "epochs": 50, 
            "batch_size": 16, 
            "shuffle": True
        }, 
        
        "statistics": {
            "graph_size": (12, 6)
        }
    }
    
    conv_ltsm_model = ConvLTSMModel(conv_ltsm_model_settings)
    conv_ltsm_model.execute()
    
    return


def lrcn_model_execute():
    lrcn_model_settings = {
        "dataset": {
            "path": DATASET_PATH, 
            "sequence_length": 150, 
            "resize_width": 64, 
            "resize_height": 64, 
            "color_channels": 3, 
            "validation_ratio": 0.2, 
            "test_ratio": 0.2,
        }, 
        
        "model": {
            "filter_counts": [16, 32, 64, 64], 
            "kernel_sizes": [(3, 3), (3, 3), (3, 3), (3, 3)], 
            "paddings": ["same", "same", "same", "same"], 
            "activations": ["relu", "relu", "relu", "relu", "softmax"], 
            "pool_sizes": [(4, 4), (4, 4), (2, 2), (2, 2)], 
            "dropouts": [0.25, 0.25, 0.25],
            "lstm_unit_count": 128,
        }, 
        
        "early_stopping": {
            "count": 1, 
            "monitors": ["val_loss"], 
            "patiences": [15], 
            "modes": ["min"], 
            "restore_best_weights": [True]
        }, 
        
        "compiling": {
            "loss": "categorical_crossentropy", 
            "learning_rate": 0.001, 
            "metrics": ["accuracy"]
        }, 
        
        "training": {
            "epochs": 80, 
            "batch_size": 16, 
            "shuffle": True
        }, 
        
        "statistics": {
            "graph_size": (12, 6)
        }
    }
    
    lrcn_model = LRCNModel(lrcn_model_settings)
    lrcn_model.execute()
    
    return


def main():
    conv_lstm_model_execute()
    lrcn_model_execute()
    
    return 0


if __name__ == "__main__":
    main()
    