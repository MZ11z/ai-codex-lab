from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path


def main() -> None:
    # 1) Load a built-in dataset (no download needed)
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 2) Split data to evaluate fairly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Pipeline: preprocessing + model
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    # 4) Train
    model.fit(X_train, y_train)

    # 5) Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    # 6) Save model artifact
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    joblib.dump(model, artifacts_dir / "model.joblib")
    print("Saved: artifacts/model.joblib")


if __name__ == "__main__":
    main()