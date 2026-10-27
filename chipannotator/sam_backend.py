from pathlib import Path
import numpy as np


class ModelError(RuntimeError):
    pass


class MobileSAM:
    """Lazy model loading. All calls are serialized by the GUI task runner."""
    def __init__(self, weights, device='auto'):
        self.weights = Path(weights)
        self.requested_device = device
        self.predictor = None
        self.image_identity = None
        self.device = None

    def load(self):
        if self.predictor is not None:
            return
        if not self.weights.is_file():
            raise ModelError(f'モデルがありません: {self.weights}\nsetup_model.bat を実行するか、設定から重みを選択してください。\n手動編集はモデルなしでも利用できます。')
        try:
            import torch
            from mobile_sam import sam_model_registry, SamPredictor
            device = self.requested_device
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                if not torch.cuda.is_available():
                    raise ModelError('CUDAを利用できません。setup_model.bat とNVIDIAドライバーを確認するかCPUを選択してください。')
                # An actual CUDA operation catches unsupported binary architectures.
                torch.ones(1, device='cuda').add_(1)
                torch.cuda.synchronize()
            model = sam_model_registry['vit_t'](checkpoint=None)
            # Do not execute arbitrary pickle code from checkpoint files.
            state = torch.load(str(self.weights), map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            model.to(device=device).eval()
            self.predictor = SamPredictor(model)
            self.device = device
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f'MobileSAMの準備に失敗しました: {e}\nsetup_model.bat / READMEのトラブル対策を確認してください。RTX 50系は対応CUDA版PyTorchが必要です。') from e

    def predict(self, image, box=None, points=None, restrict=None):
        try:
            self.load()
            import torch
            # Keep a strong reference so image identities cannot be recycled.
            if self.image_identity is not image:
                self.predictor.set_image(image)
                self.image_identity = image
            points = points or []
            coords = np.array([p[:2] for p in points], np.float32) if points else None
            labels = np.array([p[2] for p in points], np.int32) if points else None
            if box is None and not points:
                raise ModelError('矩形または正負点を指定してください')
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=coords, point_labels=labels,
                    box=np.array(box, np.float32) if box is not None else None,
                    multimask_output=True)
            mask = masks[int(np.argmax(scores))].astype(np.uint8)
            if restrict is not None:
                mask &= (restrict > 0).astype(np.uint8)
            if not mask.any():
                raise ModelError('SAMの結果が空です。矩形・正負点を調整してください。')
            return mask
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f'SAM推論に失敗しました: {e}\nメモリ不足時は画像を小さくするかCPUへ切り替えてください。') from e
