import os
import sys
import torch
import librosa
import numpy as np
import scipy
import tgt
import json
from tqdm import tqdm
from functools import partial
from distutils.util import strtobool

sys.path.append('./utils/')
import laugh_segmenter
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils

class LaughterDetector:
    def __init__(self, 
                model_path='checkpoints/in_use/resnet_with_augmentation',
                config_name='resnet_with_augmentation',
                threshold=0.5,
                min_length=0.2,
                sample_rate=8000):
        """
        Initialize the LaughterDetector with model and configuration parameters.
        """
        self.model_path = model_path
        self.config = configs.CONFIG_MAP[config_name]
        self.threshold = float(threshold)
        self.min_length = float(min_length)
        self.sample_rate = sample_rate
        self.device = self._setup_device()
        self.model = self._setup_model()
        self._debug_cuda_availability() 

    def _debug_cuda_availability(self):
        """
        Debug CUDA availability and device properties
        """
        print("\n=== CUDA Debug Information ===")
        print(f"CUDA is available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nDevice {i}: {props.name}")
                print(f"  Compute capability: {props.major}.{props.minor}")
                print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  Current device memory usage: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        
            print(f"\nCurrent device: {torch.cuda.current_device()}")
            print(f"Default device: {self.device}")
            
            # モデルのデバイス配置を確認
            if hasattr(self, 'model'):
                print("\nModel device check:")
                if next(self.model.parameters(), None) is not None:
                    print(f"Model device: {next(self.model.parameters()).device}")
                    print(f"Model memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                else:
                    print("Model has no parameters")
            else:
                print("\nModel not initialized yet")
        
        print("\nPyTorch version:", torch.__version__)
        print("===========================\n")

    def _setup_device(self):
        """
        Set up the computation device (GPU if available, otherwise CPU) with detailed logging
        """
        if torch.cuda.is_available():
            try:
                device = torch.device('cuda')
                # GPUメモリの初期状態を確認
                print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"Initial GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                
                # テストテンソルでGPUが実際に使えるか確認
                test_tensor = torch.tensor([1.0], device=device)
                print("Successfully created test tensor on GPU")
                
                print("GPU is available and working. Using CUDA.")
                return device
            except Exception as e:
                print(f"Error initializing CUDA device: {str(e)}")
                print("Falling back to CPU.")
                return torch.device('cpu')
        else:
            print("GPU is not available. Using CPU.")
            return torch.device('cpu')

    def _setup_model(self):
        """
        Set up and load the model with device placement verification
        """
        model = self.config['model'](
            dropout_rate=0.0, 
            linear_layer_size=self.config['linear_layer_size'], 
            filter_sizes=self.config['filter_sizes']
        )
        
        # モデルを明示的にデバイスに移動
        model = model.to(self.device)
        
        if os.path.exists(self.model_path):
            checkpoint_path = os.path.join(self.model_path, 'best.pth.tar')
            try:
                checkpoint = torch.load(
                    checkpoint_path, 
                    map_location=self.device,
                    weights_only=False
                )
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                
                # モデルのデバイス配置を確認
                print(f"\nModel device verification:")
                print(f"Model parameters device: {next(model.parameters()).device}")
                
                # GPUメモリ使用状況を確認（GPUの場合）
                if self.device.type == 'cuda':
                    print(f"GPU memory after model load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
        else:
            raise Exception(f"Model checkpoint not found at {self.model_path}")

        return model

    def process_audio(self, 
                     audio_path, 
                     output_dir=None, 
                     save_to_audio_files=True, 
                     save_to_textgrid=False):
        """
        Process audio file to detect and segment laughter.
        """
        # Setup inference dataset
        inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
            audio_path=audio_path, 
            feature_fn=self.config['feature_fn'], 
            sr=self.sample_rate
        )

        collate_fn = partial(
            audio_utils.pad_sequences_with_labels,
            expand_channel_dim=self.config['expand_channel_dim']
        )

        inference_generator = torch.utils.data.DataLoader(
            inference_dataset, 
            num_workers=4, 
            batch_size=8, 
            shuffle=False, 
            collate_fn=collate_fn,
            persistent_workers=True
        )

        # Make predictions
        probs = self._get_predictions(inference_generator)
        
        # Process predictions
        file_length = audio_utils.get_audio_length(audio_path)
        fps = len(probs) / float(file_length)
        probs = laugh_segmenter.lowpass(probs)
        instances = laugh_segmenter.get_laughter_instances(
            probs, 
            threshold=self.threshold, 
            min_length=self.min_length, 
            fps=fps
        )

        print(f"\nFound {len(instances)} laughs.")

        if len(instances) > 0:
            self._save_results(
                instances, 
                audio_path, 
                output_dir, 
                save_to_audio_files, 
                save_to_textgrid
            )

        return instances

    def _get_predictions(self, inference_generator):
        """
        Get predictions from the model.
        """
        probs = []
        with torch.no_grad():
            for model_inputs, _ in tqdm(inference_generator):
                x = torch.from_numpy(model_inputs).float().to(self.device)
                preds = self.model(x).cpu().detach().numpy().squeeze()
                if len(preds.shape) == 0:
                    preds = [float(preds)]
                else:
                    preds = list(preds)
                probs += preds
        return np.array(probs)

    def _save_results(self, instances, audio_path, output_dir, save_to_audio_files, save_to_textgrid):
        """
        Save results to audio files, TextGrid, and JSON format.
        """
        if not output_dir:
            raise Exception("Need to specify an output directory to save files")

        os.makedirs(output_dir, exist_ok=True)
        results = {
            "input_file": audio_path,
            "total_laughs": len(instances),
            "laughs": []
        }
        
        full_res_y, full_res_sr = librosa.load(audio_path, sr=44100)
        
        if save_to_audio_files:
            wav_paths = self._save_audio_segments(
                instances, 
                full_res_y, 
                full_res_sr, 
                output_dir
            )
            # 各笑い声セグメントの情報をJSONに追加
            for i, (instance, wav_path) in enumerate(zip(instances, wav_paths)):
                laugh_info = {
                    "id": i,
                    "start_time": float(instance[0]),
                    "end_time": float(instance[1]),
                    "duration": float(instance[1] - instance[0]),
                    "audio_file": wav_path
                }
                results["laughs"].append(laugh_info)
            print(laugh_segmenter.format_outputs(instances, wav_paths))

        if save_to_textgrid:
            textgrid_path = self._save_textgrid(instances, audio_path, output_dir)
            results["textgrid_file"] = textgrid_path

        # 結果をJSONファイルとして保存
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(output_dir, f"{base_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f'Saved detection results to {json_path}')

        return results

    def _save_audio_segments(self, instances, full_res_y, full_res_sr, output_dir):
        """
        Save individual audio segments for each detected laugh.
        """
        wav_paths = []
        maxv = np.iinfo(np.int16).max

        for index, instance in enumerate(instances):
            laughs = laugh_segmenter.cut_laughter_segments(
                [instance], 
                full_res_y, 
                full_res_sr
            )
            wav_path = os.path.join(output_dir, f"laugh_{index}.wav")
            scipy.io.wavfile.write(
                wav_path, 
                full_res_sr, 
                (laughs * maxv).astype(np.int16)
            )
            wav_paths.append(wav_path)

        return wav_paths

    def _save_textgrid(self, instances, audio_path, output_dir):
        """
        Save laughter segments to TextGrid format.
        Returns the path to the saved TextGrid file.
        """
        laughs = [{'start': i[0], 'end': i[1]} for i in instances]
        tg = tgt.TextGrid()
        laughs_tier = tgt.IntervalTier(
            name='laughter', 
            objects=[tgt.Interval(l['start'], l['end'], 'laugh') for l in laughs]
        )
        tg.add_tier(laughs_tier)
        
        fname = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{fname}_laughter.TextGrid")
        tgt.write_to_file(tg, output_path)
        print(f'Saved laughter segments in {output_path}')
        return output_path


import argparse
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/in_use/resnet_with_augmentation')
    parser.add_argument('--config', type=str, 
                       default='resnet_with_augmentation')
    parser.add_argument('--threshold', type=str, default='0.5')
    parser.add_argument('--min_length', type=str, default='0.2')
    parser.add_argument('--input_audio_file', required=True, type=str)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_to_audio_files', type=str, default='True')
    parser.add_argument('--save_to_textgrid', type=str, default='False')
    return parser.parse_args()

def main():
    args = parse_args()
    
    detector = LaughterDetector(
        model_path=args.model_path,
        config_name=args.config,
        threshold=float(args.threshold),
        min_length=float(args.min_length)
    )
    
    detector.process_audio(
        audio_path=args.input_audio_file,
        output_dir=args.output_dir,
        save_to_audio_files=bool(strtobool(args.save_to_audio_files)),
        save_to_textgrid=bool(strtobool(args.save_to_textgrid))
    )

if __name__ == '__main__':
    main()