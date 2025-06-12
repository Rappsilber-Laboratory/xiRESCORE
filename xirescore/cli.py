import argparse
import yaml
import ast
import logging
import os
import sys

import logging_loki

from xirescore.XiRescore import XiRescore
import xirescore
from xirescore._gui import create_gui

logger = logging.getLogger(__name__)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Rescoring crosslinked-peptide identifications.')

    # Define CLI arguments
    parser.add_argument('-i', action='store', dest='input_path', help='input path',
                        type=str, required=False)
    parser.add_argument('-o', action='store', dest='output_path', help='output path',
                        type=str, required=False)
    parser.add_argument('-c', action='store', dest='config_file', help='config file',
                        type=str, required=False)
    parser.add_argument('-C', action='store', dest='config_string', help='config test',
                        type=str, required=False)
    parser.add_argument('--loki', action='store', dest='loki', help='Loki server address',
                        type=str, required=False)
    parser.add_argument('--loki-job-id', action='store', dest='loki_job_id', help='Loki job ID',
                        default="", type=str, required=False)
    parser.add_argument('--version', action='store_true', dest='print_version', help='print version')

    # Parse arguments
    args = parser.parse_args()

    if args.print_version:
        print(xirescore.__version__)
        os._exit(os.EX_OK)

    if (args.input_path is None) or (args.output_path is None):
        create_gui()
    else:
        run_headless(args)


def run_headless(args):
    logger = logging.getLogger('xirescore')

    # Configure Loki logger
    if args.loki is None:
        logging.basicConfig()
        logger.setLevel(logging.DEBUG)
    else:
        handler = logging_loki.LokiHandler(
            url=args.loki,
            tags={
                "application": "xirescore",
                "job_id": args.loki_job_id
            },
            version="1",
        )
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

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
    rescorer.run()
    logger.info("Done.")


if __name__ == "__main__":
    main()
