from detector import LaughterDetector
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