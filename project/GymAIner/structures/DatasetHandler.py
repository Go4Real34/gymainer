import os
import cv2
import tqdm
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DatasetHandler:
    def __init__(self, dataset_path, validation_ratio, test_ratio):
        self.RESIZE_WIDTH = 64
        self.RESIZE_HEIGHT = 64
        
        self.dataset_path = dataset_path
        self.VALIDATION_RATIO = validation_ratio
        self.TEST_RATIO = test_ratio
        
        self.video_paths, self.labels, self.videos = self.read_dataset()
        
        self.X_train, X_validation, self.Y_train, Y_validation = train_test_split(self.videos, self.labels, test_size=self.VALIDATION_RATIO)
        self.X_validation, self.X_test, self.Y_validation, self.Y_test = train_test_split(X_validation, Y_validation, test_size=self.TEST_RATIO)
        
        return
    
    def read_dataset(self):
        paths = {}
        categories = os.listdir(self.dataset_path)
        for category in categories:
            if category not in paths:
                paths[category] = []
                
            folder_path = os.path.join(self.dataset_path, category)
            numpy_array_video_frames = os.listdir(folder_path)
            for video in numpy_array_video_frames:
                video_path = os.path.join(folder_path, video)
                paths[category].append(video_path)
                
        numpy_array_labels = []
        numpy_array_video_frames = []

        total_videos = sum([len(v) for k, v in paths.items()])
        with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]",
                       desc="Saving videos...", unit=" videos", total=total_videos) as pbar_full:
            label_count = len(paths)
            
            with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]", 
                            desc="Reading folders...", unit=" folders", total=label_count) as pbar_folders:
                for category, video_path_list in paths.items():
                    pbar_folders.set_description(f"Reading folder: {category}")
                    
                    video_count = len(video_path_list)
                    with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]", 
                                   desc="Reading videos...", unit= " videos", total=video_count, leave=False) as pbar_videos:
                        for video_path in video_path_list:
                            dataset_tag, main_folder, category, video_name = video_path.split('\\')
                            pbar_videos.set_description(f"Reading video: {video_name}")
                            
                            video_frames = self.read_video(video_path)
                            numpy_array_labels.append(category)
                            numpy_array_video_frames.append(video_frames)
                            
                            pbar_videos.update(1)
                            pbar_full.update(1)
                                
                    pbar_folders.update(1)

        numpy_array_labels = np.asarray(numpy_array_labels)
        categorical_labels = to_categorical(numpy_array_labels)

        numpy_array_video_frames = np.asarray(numpy_array_video_frames)
        
        return paths, categorical_labels, numpy_array_video_frames   

    def read_video(self, video_path):
        frames = []
        video_reader = cv2.VideoCapture(video_path)
        if not video_reader.isOpened():
            print("Error: Could not open video.")
            return
        
        frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]", 
                       desc="Reading frames...", unit=" frames", total=frame_count, leave=False) as pbar_frames:
            for frame_index in range(frame_count):
                success, frame = video_reader.read()
            
                if frame is None:
                    break
            
                if not success:
                    print("Error: Could not read frame.")
                    break
            
                processed_frame = self.process_frame(frame)
                frames.append(processed_frame)
                pbar_frames.update(1)
        
            video_reader.release()
        
        return frames
    
    def process_frame(self, frame):
        resized_frame = cv2.resize(frame, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
        gray_scaled_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_scaled_image)
        normalized_frame = equalized_image / 255.0
        return normalized_frame
    