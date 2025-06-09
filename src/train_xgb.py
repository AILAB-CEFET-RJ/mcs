# train_xgb.py
import argparse
from models import get_xgb_poisson
from utils import plot_learning_curve, smape
from data_utils import load_and_prepare_dataset
from evaluation import evaluate_model_basic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_dataset(args.dataset)
    model = get_xgb_poisson(seed=args.seed)

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    evaluate_model_basic(model, X_train, y_train, X_val, y_val, X_test, y_test, args.outdir)