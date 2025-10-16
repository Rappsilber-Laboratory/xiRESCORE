import argparse
import pickle
import sys
import os

import yaml
import ast
import logging

import logging_loki

from xirescore.XiRescore import XiRescore
import xirescore
from xirescore._gui import create_gui

logger = logging.getLogger(__name__)

def main():
    # Fixes regarding multiprocessing and pyinstaller
    import multiprocessing
    multiprocessing.freeze_support()

    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    # Create argument parser
    parser = argparse.ArgumentParser(description='Rescoring crosslinked-peptide identifications.')

    # Define CLI arguments
    parser.add_argument('-i', action='store', dest='input_path', help='input path',
                        type=str, required=False)
    parser.add_argument('-o', action='store', dest='output_path', help='output path',
                        type=str, required=False)
    parser.add_argument('-c', action='store', dest='config_file', help='config file',
                        type=str, required=False)
    parser.add_argument('-C', action='store', dest='config_string', help='config string',
                        type=str, required=False)
    parser.add_argument('-m', action='store', dest='model_input', help='pre-trained input model path',
                        type=str, required=False)
    parser.add_argument('-M', action='store', dest='model_output', help='model export path',
                        type=str, required=False)
    parser.add_argument('--debug', action='store_true', dest='debug', help='Debug logging')
    parser.add_argument('--version', action='store_true', dest='print_version', help='print version')

    # Parse arguments
    args = parser.parse_args()

    if args.print_version:
        print(xirescore.__version__)
        sys.exit(0)

    if (args.input_path is None) or (args.output_path is None):
        create_gui()
    else:
        run_headless(args)


def run_headless(args):
    logger = logging.getLogger('xirescore')

    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Load config
    if args.config_file is not None:
        with open(args.config_file, 'r') as file:
            options = yaml.safe_load(file)
    elif args.config_string is not None:
        options = ast.literal_eval(args.config_string)
    else:
        options = dict()

    # Configure stdout logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Initialize XiRescore
    rescorer = XiRescore(
        input_path=args.input_path,
        output_path=args.output_path,
        options=options,
    )

    # Use pre-trained model if defined
    if isinstance(args.model_input, str):
        with open(args.model_input, 'rb') as f:
            model = pickle.load(f)
        rescorer.pca = model['pca']
        rescorer.imputer = model['imputer']
        rescorer.scaler = model['scaler']
        rescorer.train_df = model['training_data']
        rescorer.train_features = model['train_features']
        rescorer.splits = model['splits']
        rescorer.models = model['models']
        rescorer.rescore()
    else:
        rescorer.run()

    if isinstance(args.model_output, str):
        logger.info(f"Storing model under {args.model_output}.")
        with open(args.model_output, 'wb') as f:
            pickle.dump(rescorer.get_rescoring_state(), f)

    logger.info("Done.")


if __name__ == "__main__":
    main()
