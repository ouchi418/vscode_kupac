#network
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
    hidden_sizes: tuple[int, ...] = (512, 256, 128, 64, 32)
    output_size: int = 10
    learning_rate: float = 0.001
    dropout_rate: float = 0.2     # 【追加】ドロップアウト率（20%の細胞を休ませる）
    batch_size: int = 128
    seed: int = 42


class SimpleMLP:
    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        
        layer_sizes = [config.input_size] + list(config.hidden_sizes) + [config.output_size]
        self.num_layers = len(layer_sizes) - 1
        
        self.params: dict[str, np.ndarray] = {}
        for i in range(1, self.num_layers + 1):
            n_in = layer_sizes[i - 1]
            n_out = layer_sizes[i]
            scale = np.sqrt(2.0 / n_in)
            self.params[f"W{i}"] = (rng.standard_normal((n_in, n_out)) * scale).astype(np.float32)
            self.params[f"b{i}"] = np.zeros(n_out, dtype=np.float32)

        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        a = x
        for i in range(1, self.num_layers):
            z = np.dot(a, self.params[f"W{i}"]) + self.params[f"b{i}"]
            a = np.maximum(0, z) # テスト時は全細胞を動かす（Dropoutなし）
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

    def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int, current_lr: float) -> float:
        # 【変更】現在の学習率(current_lr)を引数として受け取るようにしました
        rng = np.random.default_rng(self.config.seed + epoch)
        indices = rng.permutation(x.shape[0])
        total_loss = 0.0
        steps = 0
        batch_size = self.config.batch_size
        dr = self.config.dropout_rate

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        for start in range(0, x.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            # --- 1. 順伝播 (Forward) ---
            activations = [x_batch]
            linears = []
            masks = [] # 【追加】各層のDropoutマスクを記憶するリスト
            
            a = x_batch
            for i in range(1, self.num_layers):
                z = np.dot(a, self.params[f"W{i}"]) + self.params[f"b{i}"]
                linears.append(z)
                a = np.maximum(0, z)
                
                # 【追加】Dropoutの処理 (Inverted Dropout)
                if dr > 0.0:
                    # 確率 (1-dr) で1、それ以外は0のマスクを作る
                    mask = (rng.random(a.shape) >= dr) / (1.0 - dr)
                    a = a * mask
                    masks.append(mask)
                else:
                    masks.append(None)
                    
                activations.append(a)
                
            logits = np.dot(a, self.params[f"W{self.num_layers}"]) + self.params[f"b{self.num_layers}"]
            probs = _softmax(logits)

            y_one_hot = _one_hot(y_batch, self.config.output_size)
            loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
            total_loss += float(loss)
            steps += 1

            # --- 2. 逆伝播 (Backward) ---
            grads: dict[str, np.ndarray] = {}
            d_logits = (probs - y_one_hot) / x_batch.shape[0]
            grads[f"W{self.num_layers}"] = np.dot(activations[-1].T, d_logits)
            grads[f"b{self.num_layers}"] = np.sum(d_logits, axis=0)
            
            d_prev = np.dot(d_logits, self.params[f"W{self.num_layers}"].T)

            for i in range(self.num_layers - 1, 0, -1):
                # 【追加】順伝播で休ませた細胞には、逆伝播でも誤差を流さない（マスクの適用）
                if dr > 0.0 and masks[i - 1] is not None:
                    d_prev = d_prev * masks[i - 1]
                    
                z = linears[i - 1]
                d_z = d_prev * (z > 0)
                
                grads[f"W{i}"] = np.dot(activations[i - 1].T, d_z)
                grads[f"b{i}"] = np.sum(d_z, axis=0)
                
                if i > 1:
                    d_prev = np.dot(d_z, self.params[f"W{i}"].T)

            # --- 3. パラメータの更新 (Adam) ---
            self.t += 1
            for key in self.params:
                self.m[key] = beta1 * self.m[key] + (1.0 - beta1) * grads[key]
                self.v[key] = beta2 * self.v[key] + (1.0 - beta2) * np.square(grads[key])
                m_hat = self.m[key] / (1.0 - beta1 ** self.t)
                v_hat = self.v[key] / (1.0 - beta2 ** self.t)
                # 【変更】引数で受け取った current_lr を使用
                self.params[key] -= current_lr * m_hat / (np.sqrt(v_hat) + eps)

        return total_loss / max(steps, 1)

    def to_state(self) -> dict[str, object]:
        return {
            "model_type": "DeepMLP_Adam_Dropout",
            "config": {
                "input_size": self.config.input_size,
                "hidden_sizes": self.config.hidden_sizes,
                "output_size": self.config.output_size,
                "learning_rate": self.config.learning_rate,
                "dropout_rate": self.config.dropout_rate,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
            },
            "params": self.params,
            "m": self.m,
            "v": self.v,
            "t": self.t
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SimpleMLP":
        config_obj = state.get("config")
        config_dict: dict[str, Any] = config_obj
        config = NetworkConfig(
            input_size=int(config_dict["input_size"]),
            hidden_sizes=tuple(config_dict.get("hidden_sizes", (128, 64))),
            output_size=int(config_dict["output_size"]),
            learning_rate=float(config_dict.get("learning_rate", 0.001)),
            dropout_rate=float(config_dict.get("dropout_rate", 0.2)),
            batch_size=int(config_dict.get("batch_size", 128)),
            seed=int(config_dict.get("seed", 42)),
        )
        model = cls(config)
        model.params = {k: v for k, v in state["params"].items()}
        if "m" in state: model.m = state["m"]
        if "v" in state: model.v = state["v"]
        if "t" in state: model.t = state["t"]
        return model