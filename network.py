from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


@dataclass
class NetworkConfig:
    input_size: int = 784
    hidden_sizes: tuple[int, ...] = (128, 64)  # 【変更】複数の層のサイズをタプルで保持
    output_size: int = 10
    learning_rate: float = 0.1
    batch_size: int = 128
    seed: int = 42


class SimpleMLP:
    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        
        # 全ての層のノード数をリスト化 (例: [784, 128, 64, 32, 10])
        layer_sizes = [config.input_size] + list(config.hidden_sizes) + [config.output_size]
        self.num_layers = len(layer_sizes) - 1
        
        self.params: dict[str, np.ndarray] = {}
        
        # ループを使って全層の重みとバイアスを初期化（Heの初期値）
        for i in range(1, self.num_layers + 1):
            n_in = layer_sizes[i - 1]
            n_out = layer_sizes[i]
            scale = np.sqrt(2.0 / n_in)  # Heの初期値
            
            self.params[f"W{i}"] = (rng.standard_normal((n_in, n_out)) * scale).astype(np.float32)
            self.params[f"b{i}"] = np.zeros(n_out, dtype=np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        a = x
        # 隠れ層の順伝播 (ReLU)
        for i in range(1, self.num_layers):
            z = np.dot(a, self.params[f"W{i}"]) + self.params[f"b{i}"]
            a = np.maximum(0, z)
            
        # 出力層の順伝播 (Softmax)
        logits = np.dot(a, self.params[f"W{self.num_layers}"]) + self.params[f"b{self.num_layers}"]
        return _softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        correct = 0
        total = x.shape[0]
        batch_size = self.config.batch_size
        for i in range(0, total, batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            pred = self.predict(x_batch)
            correct += int(np.sum(pred == y_batch))
        return float(correct) / float(total)

    def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int) -> float:
        rng = np.random.default_rng(self.config.seed + epoch)
        indices = rng.permutation(x.shape[0])
        total_loss = 0.0
        steps = 0
        batch_size = self.config.batch_size
        lr = self.config.learning_rate

        for start in range(0, x.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            # --- 1. 順伝播 (Forward) ---
            activations = [x_batch]  # 各層の出力(a)を保存
            linears = []             # 各層の活性化関数を通す前の値(z)を保存
            
            a = x_batch
            for i in range(1, self.num_layers):
                z = np.dot(a, self.params[f"W{i}"]) + self.params[f"b{i}"]
                linears.append(z)
                a = np.maximum(0, z)  # ReLU
                activations.append(a)
                
            # 出力層
            logits = np.dot(a, self.params[f"W{self.num_layers}"]) + self.params[f"b{self.num_layers}"]
            probs = _softmax(logits)

            # 誤差の計算
            y_one_hot = _one_hot(y_batch, self.config.output_size)
            loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
            total_loss += float(loss)
            steps += 1

            # --- 2. 逆伝播 (Backward) ---
            grads: dict[str, np.ndarray] = {}
            
            # 出力層の勾配
            d_logits = (probs - y_one_hot) / x_batch.shape[0]
            grads[f"W{self.num_layers}"] = np.dot(activations[-1].T, d_logits)
            grads[f"b{self.num_layers}"] = np.sum(d_logits, axis=0)
            
            # 次の層（後ろの層）に伝える誤差
            d_prev = np.dot(d_logits, self.params[f"W{self.num_layers}"].T)

            # 隠れ層の勾配（後ろから前へループ）
            for i in range(self.num_layers - 1, 0, -1):
                z = linears[i - 1]
                # ReLUの微分
                d_z = d_prev * (z > 0)
                
                grads[f"W{i}"] = np.dot(activations[i - 1].T, d_z)
                grads[f"b{i}"] = np.sum(d_z, axis=0)
                
                if i > 1:
                    d_prev = np.dot(d_z, self.params[f"W{i}"].T)

            # --- 3. パラメータの更新 ---
            for key in self.params:
                self.params[key] -= lr * grads[key].astype(np.float32)

        return total_loss / max(steps, 1)

    def to_state(self) -> dict[str, object]:
        return {
            "model_type": "DeepMLP",
            "config": {
                "input_size": self.config.input_size,
                "hidden_sizes": self.config.hidden_sizes,
                "output_size": self.config.output_size,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
            },
            "params": self.params,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SimpleMLP":
        config_obj = state.get("config")
        if not isinstance(config_obj, dict):
            raise ValueError("Invalid state: 'config' must be a dict")
        config_dict: dict[str, Any] = config_obj

        config = NetworkConfig(
            input_size=int(config_dict["input_size"]),
            hidden_sizes=tuple(config_dict.get("hidden_sizes", (64,))),
            output_size=int(config_dict["output_size"]),
            learning_rate=float(config_dict.get("learning_rate", 0.1)),
            batch_size=int(config_dict.get("batch_size", 128)),
            seed=int(config_dict.get("seed", 42)),
        )

        params_obj = state.get("params")
        if not isinstance(params_obj, dict):
            raise ValueError("Invalid state: 'params' must be a dict")
        
        model = cls(config)
        model.params = {k: v for k, v in params_obj.items()}
        return model