import argparse
# 1. Importer spesifikt det du trenger
from src.sibr_market_training import Clean, Train, Predict
from sibr_module import Logger
import logging
from dotenv import load_dotenv
load_dotenv(verbose=True)

# 2. Definer konstanter for å unngå "magiske strenger" og repetisjon
SUPPORTED_DATASETS = ['cars', 'homes', 'rentals']
TRAIN_PREDICT_DATASETS = ['cars', 'homes']


def main():
    """Hovedfunksjon for å kjøre skriptet."""
    parser = argparse.ArgumentParser(description="Run data processing and training pipelines for SIBR Market.")

    # 3. Bruk action='store_true' for boolean-flagg
    parser.add_argument('--dataset', type=str, help=f'Dataset name. Choose from: {SUPPORTED_DATASETS}')
    parser.add_argument('--task', type=str, help="Specific task to run: 'clean', 'pre_processed', 'train', 'predict'")

    # Flagg for å styre kjøring
    parser.add_argument('--run_all', action='store_true', help='Run all tasks for cars and homes except training. To also run training, use --task train or --run_train.')
    parser.add_argument('--run_clean', action='store_true', help=f'Run cleaning for all datasets: {SUPPORTED_DATASETS}')
    parser.add_argument('--run_train', action='store_true', help='Run training for a specific dataset.')
    parser.add_argument('--run_predict', action='store_true', help='Run prediction for a specific dataset.')
    parser.add_argument('--log-level', type = str, default='DEBUG' ,help='Loglevel')
    parser.add_argument('--cloud-logging', action='store_true', help='Enables cloud logging.')

    # Flagg for datalagring
    parser.add_argument('--no_save',
                        action='store_true',
                        help='Disable saving data to BigQuery. (Enabled by default)')
    parser.add_argument('--replace', action='store_true', help='Replace existing data in BigQuery.')

    args = parser.parse_args()

    #global logger
    logger = Logger(log_name='main',enable_cloud_logging=args.cloud_logging)
    logger.log_level = "DEBUG"

    # Inverterer flagget for enklere bruk i koden
    save_to_bq = not args.no_save

    if args.log_level != 'DEBUG':
        log_level = getattr(logging, args.log_level.upper())
        for handler in logger._logger.handlers:
            handler.setLevel(log_level)

    def run_cleaning_pipeline(dataset: str, save_to_bq: bool, replace: bool):
        """Kjører rense- og preprosesseringssteg for et gitt datasett."""
        logger.info(f"--- Running cleaning pipeline for dataset: {dataset} ---")
        cleaner = Clean(dataset=dataset,logger = logger)

        # Kjør rensing
        # La df være None hvis det feiler, eller en DataFrame hvis det lykkes
        df = cleaner.run(task='clean', save_to_bq=save_to_bq, replace=replace)

        # Kjør preprosessering. Bruk den rensede df-en hvis den finnes.
        if df is not None:
            logger.info("Using dataframe from previous step for pre-processing.")
            cleaner.run(task='pre_processed', df=df, save_to_bq=save_to_bq, replace=replace)
        else:
            logger.info("No dataframe from previous step. Running pre-processing from source.")
            cleaner.run(task='pre_processed', save_to_bq=save_to_bq, replace=replace)
        logger.info(f"--- Finished cleaning pipeline for {dataset} ---")


    # 4. Bruk en if/elif/else-struktur for å unngå overlappende logikk
    if args.run_all:
        logger.info("=== Running all tasks for cars and homes ===")
        for dataset in TRAIN_PREDICT_DATASETS:
            run_cleaning_pipeline(dataset, save_to_bq, args.replace)

            if args.task == 'train' or args.run_train:
                logger.info(f"--- Running training for: {dataset} ---")
                train = Train(dataset=dataset, logger=logger)
                train.run()

            logger.info(f"--- Running prediction for: {dataset} ---")
            predict = Predict(dataset=dataset, logger=logger)
            predict.run()

    elif args.run_clean:
        if args.dataset and args.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"For cleaning, --dataset must be one of: {SUPPORTED_DATASETS}")
        if args.dataset:
            logger.info(f"=== Cleaning for: {args.dataset} ===")
            run_cleaning_pipeline(args.dataset, save_to_bq, args.replace)
        else:
            logger.info(f"=== Running cleaning for all supported datasets ===")
            for dataset in SUPPORTED_DATASETS:
                run_cleaning_pipeline(dataset, save_to_bq, args.replace)

    elif args.run_train:
        if args.dataset and args.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"For training, --dataset must be one of: {SUPPORTED_DATASETS}")
        if args.dataset:
                logger.info(f"=== Running training for: {args.dataset} ===")
                train = Train(dataset=args.dataset, logger=logger)
                train.run()
        else:
            logger.info("=== Running training for all datasets ===")
            for dataset in SUPPORTED_DATASETS:
                logger.info(f"--- Running training for: {dataset} ---")
                train = Train(dataset=dataset, logger=logger)
                train.run()

    elif args.run_predict:
        if args.dataset and args.dataset not in TRAIN_PREDICT_DATASETS:
            raise ValueError(f"For prediction, --dataset must be one of: {TRAIN_PREDICT_DATASETS}")
        if args.dataset:
            logger.info(f"=== Running prediction for: {args.dataset} ===")
            predict = Predict(dataset=args.dataset, logger=logger)
            predict.run()
        else:
            for dataset in TRAIN_PREDICT_DATASETS:
                logger.info(f"--- Running prediction for: {dataset} ---")
                predict = Predict(dataset=dataset, logger=logger)
                predict.run()

    elif args.dataset and args.task:
        if args.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Dataset must be one of: {SUPPORTED_DATASETS}")

        logger.info(f"=== Running task '{args.task}' for dataset '{args.dataset}' ===")
        if args.task in ['clean', 'pre_processed']:
            cleaner = Clean(dataset=args.dataset, logger=logger)
            cleaner.run(task=args.task, save_to_bq=save_to_bq, replace=args.replace)
        elif args.task == 'train':
            train = Train(dataset=args.dataset, logger=logger)
            train.run()
        elif args.task == 'predict':
            predict = Predict(dataset=args.dataset, logger=logger)
            predict.run()
        else:
            raise ValueError("Task must be one of: 'clean', 'pre_processed', 'train', 'predict'.")

    else:
        logger.info("No specific run command provided. Use --help for options.")


if __name__ == "__main__":
    main()