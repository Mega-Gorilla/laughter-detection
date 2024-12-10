import os
import sys
import torch
import librosa
import numpy as np
import scipy
import tgt
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

    def _setup_device(self):
        """
        Set up the computation device (GPU if available, otherwise CPU).
        """
        if torch.cuda.is_available():
            print("GPU is available. Using CUDA.")
            return torch.device('cuda')
        else:
            print("GPU is not available. Using CPU.")
            return torch.device('cpu')

    def _setup_model(self):
        """
        Set up and load the model.
        """
        model = self.config['model'](
            dropout_rate=0.0, 
            linear_layer_size=self.config['linear_layer_size'], 
            filter_sizes=self.config['filter_sizes']
        )
        model.set_device(self.device)

        if os.path.exists(self.model_path):
            checkpoint_path = os.path.join(self.model_path, 'best.pth.tar')
            # セーフリストの問題を回避するために直接weights_only=Falseを使用
            checkpoint = torch.load(
                checkpoint_path, 
                map_location=self.device,
                weights_only=False  # 信頼できるモデルの場合(セキュリテリスクあり)
            )
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
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
            collate_fn=collate_fn
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
        Save results to audio files and/or TextGrid.
        """
        if not output_dir:
            raise Exception("Need to specify an output directory to save files")

        os.makedirs(output_dir, exist_ok=True)
        full_res_y, full_res_sr = librosa.load(audio_path, sr=44100)
        
        if save_to_audio_files:
            wav_paths = self._save_audio_segments(
                instances, 
                full_res_y, 
                full_res_sr, 
                output_dir
            )
            print(laugh_segmenter.format_outputs(instances, wav_paths))

        if save_to_textgrid:
            self._save_textgrid(instances, audio_path, output_dir)

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