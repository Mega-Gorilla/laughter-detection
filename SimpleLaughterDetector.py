import os
from dataclasses import dataclass
from typing import List, Optional, Dict
import json
from .detector import LaughterDetector

@dataclass
class LaughSegment:
    """笑い声セグメントの情報を保持するデータクラス"""
    start_time: float
    end_time: float
    duration: float
    audio_file: Optional[str] = None

class SimpleLaughterDetector:
    """
    笑い声検出 ラッパークラス
    """
    def __init__(
        self,
        model_path: str = 'checkpoints/in_use/resnet_with_augmentation',
        threshold: float = 0.5,
        min_length: float = 0.2,
        use_gpu: bool = True
    ):
        """
        Parameters
        ----------
        model_path : str
            学習済みモデルのパス
        threshold : float
            笑い声検出の閾値 (0.0 ~ 1.0)
        min_length : float
            検出する笑い声の最小長さ（秒）
        use_gpu : bool
            GPUを使用するかどうか
        """
        
        self.detector = LaughterDetector(
            model_path=model_path,
            config_name='resnet_with_augmentation',
            threshold=threshold,
            min_length=min_length
        )
        
        # GPUが利用不可の場合は警告を出す
        if use_gpu and not self.detector.device.type == 'cuda':
            print("Warning: GPU requested but not available. Using CPU instead.")

    def detect_from_file(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        save_audio: bool = False,
        save_textgrid: bool = False
    ) -> List[LaughSegment]:
        """
        音声ファイルから笑い声を検出する

        Parameters
        ----------
        audio_path : str
            解析する音声ファイルのパス
        output_dir : str, optional
            結果を保存するディレクトリ
        save_audio : bool
            個別の音声ファイルとして保存するかどうか
        save_textgrid : bool
            TextGridファイルとして保存するかどうか

        Returns
        -------
        List[LaughSegment]
            検出された笑い声セグメントのリスト
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 検出実行
        results = self.detector.process_audio(
            audio_path=audio_path,
            output_dir=output_dir,
            save_to_audio_files=save_audio,
            save_to_textgrid=save_textgrid
        )

        # 結果をLaughSegmentのリストに変換
        laugh_segments = []
        for laugh in results:
            start_time, end_time = laugh
            duration = end_time - start_time
            segment = LaughSegment(
                start_time=float(start_time),
                end_time=float(end_time),
                duration=float(duration)
            )
            laugh_segments.append(segment)

        return laugh_segments

    def save_results(
        self,
        laugh_segments: List[LaughSegment],
        output_path: str
    ) -> None:
        """
        検出結果をJSONファイルとして保存する

        Parameters
        ----------
        laugh_segments : List[LaughSegment]
            検出された笑い声セグメントのリスト
        output_path : str
            保存先のJSONファイルパス
        """
        results = {
            "total_laughs": len(laugh_segments),
            "laughs": [
                {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "duration": segment.duration,
                    "audio_file": segment.audio_file
                }
                for segment in laugh_segments
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

# 使用例
if __name__ == "__main__":
    # 検出器のインスタンス化
    detector = SimpleLaughterDetector(
        threshold=0.6,
        min_length=0.3,
        use_gpu=True
    )
    
    # 笑い声の検出
    laugh_segments = detector.detect_from_file(
        audio_path="sample.wav",
        output_dir="output",
        save_audio=True
    )
    
    # 結果の表示と保存
    print(f"検出された笑い声の数: {len(laugh_segments)}")
    for i, segment in enumerate(laugh_segments):
        print(f"笑い声 {i + 1}: {segment.start_time:.2f}秒 - {segment.end_time:.2f}秒 "
              f"(長さ: {segment.duration:.2f}秒)")
    
    detector.save_results(laugh_segments, "output/results.json")