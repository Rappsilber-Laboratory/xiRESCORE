import logging
import random
import yaml

import numpy as np
import polars as pl

from xirescore.XiRescore import XiRescore


def test_convergence_error():
    with open('tests/fixtures/config_scout.yaml', 'r') as file:
        options = yaml.safe_load(file)
    random.seed(0)
    np.random.seed(0)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info('Start full DF rescoring test')

    options['rescoring']['model_class'] = 'neural_network'
    options['rescoring']['model_name'] = 'MLPClassifier'
    options['rescoring']['model_params'] = dict()
    options['rescoring']['model_params']['hidden_layer_sizes'] = [
        [500, 500]
    ]
    options['rescoring']['model_params']['alpha'] = [1e-10]
    options['rescoring']['model_params']['learning_rate_init'] = [1.0]
    options['rescoring']['model_params']['max_iter'] = [10]
    options['rescoring']['model_params']['solver'] = ['adam']
    options['rescoring']['model_params']['activation'] = ['tanh']

    rescorer = XiRescore(
        input_path='tests/fixtures/scout.csv',
        options=options,
    )
    rescorer.run()
