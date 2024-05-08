class ModelSettings:
    def __init__(self, 
                 dataset_path, 
                 sequence_length, 
                 validation_ratio, 
                 test_ratio, 
                 filter_counts, 
                 kernel_sizes, 
                 activations, 
                 data_formats, 
                 recurrent_dropouts, 
                 return_sequences, 
                 pool_sizes, 
                 paddings, 
                 dropouts, 
                 label_count):
        
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.filter_counts = filter_counts
        self.kernel_sizes = kernel_sizes
        self.activations = activations
        self.data_formats = data_formats
        self.recurrent_dropouts = recurrent_dropouts
        self.return_sequences = return_sequences
        self.pool_sizes = pool_sizes
        self.paddings = paddings
        self.dropouts = dropouts
        self.label_count = label_count
        
        return
    
    def get_dataset_path(self):
        return self.dataset_path
    
    def get_sequence_length(self):
        return self.sequence_length
    
    def get_validation_ratio(self):
        return self.validation_ratio
    
    def get_test_ratio(self):
        return self.test_ratio
    
    def get_filter_count(self, layer):
        return self.filter_counts[layer]

    def get_kernel_size(self, layer):
        return self.kernel_sizes[layer]
    
    def get_activation(self, layer):
        return self.activations[layer]
    
    def get_data_format(self, layer):
        return self.data_formats[layer]

    def get_recurrent_dropout(self, layer):
        return self.recurrent_dropouts[layer]
    
    def get_return_sequence(self, layer):
        return self.return_sequences[layer]
    
    def get_pool_size(self, layer):
        return self.pool_sizes[layer]
    
    def get_paddings(self, layer):
        return self.paddings[layer]
    
    def get_dropout(self, layer):
        return self.dropouts[layer]
    
    def get_label_count(self, layer):
        return self.label_count
    