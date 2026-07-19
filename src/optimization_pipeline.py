import argparse

from optimization.experiment_controller import ExperimentController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiments.yaml")
    parser.add_argument("--dataset", default=None, help="Nome exato, por exemplo RN_WEEKLY")
    parser.add_argument("--model", choices=["poisson", "zip", "rf"], default=None)
    args = parser.parse_args()

    controller = ExperimentController(args.config)
    controller.run(dataset_filter=args.dataset, model_filter=args.model)


if __name__ == "__main__":
    main()
