import pickle
from pathlib import Path

from load_fashion_mnist import load_train_data
from network import NetworkConfig, SimpleMLP

OUTPUT_PATH = Path("sample_weight.pkl")
EPOCHS = 100
# 【変更】複数の層のサイズをタプルで指定する形に変更しました
HIDDEN_SIZES = (512, 256, 128, 64, 32)  # 例：1層目128個、2層目64個、3層目32個
LEARNING_RATE = 0.05
BATCH_SIZE = 128
SEED = 42

def main() -> int:
    (x_train, t_train), (x_valid, t_valid) = load_train_data()

    model = SimpleMLP(
        NetworkConfig(
            input_size=x_train.shape[1],
            hidden_sizes=HIDDEN_SIZES,  # 【変更】リスト（タプル）を渡す
            output_size=10,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
    )

    for epoch in range(1, EPOCHS + 1):
        loss = model.train_epoch(x_train, t_train, epoch=epoch)
        train_acc = model.evaluate_accuracy(x_train, t_train)
        valid_acc = model.evaluate_accuracy(x_valid, t_valid)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={loss:.4f} train_acc={train_acc:.4f} valid_acc={valid_acc:.4f}"
        )

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(model.to_state(), f)

    print(f"Saved model: {OUTPUT_PATH.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())