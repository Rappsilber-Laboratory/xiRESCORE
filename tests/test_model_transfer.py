import sys
import pytest
import logging
import tempfile
import subprocess
import os

@pytest.mark.cli
def test_model_transfer():
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info('Start full CSV rescoring test')
        logger.info(f'Write results to {tmpdirname}')

        options = {
            'input': {
                'columns': {
                    'features': [
                        'match_score',
                        'better_score',
                        'worse_score',
                        'useless_score_uni',
                        'useless_score_norm',
                        'conditional_score',
                    ],
                    'score': 'match_score'
                },
            },
            'rescoring': {
                'spectra_batch_size': 25_000,  # Rescore in 4 batches
            }
        }

        python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
        result = subprocess.run(
            [
                f'python{python_version}',
                'xirescore',
                '-i', './tests/fixtures/test_data.parquet',
                '-o', f'{tmpdirname}/result.parquet',
                '-C', str(options),
                '-M', f'{tmpdirname}/model.p',
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"
        assert os.path.exists(f'{tmpdirname}/result.parquet'), f"Output file was not created."
        assert os.path.exists(f'{tmpdirname}/model.p'), f"Output model was not created."

        result2 = subprocess.run(
            [
                f'python{python_version}',
                'xirescore',
                '-i', './tests/fixtures/test_data.parquet',
                '-o', f'{tmpdirname}/result2.parquet',
                '-C', str(options),
                '-m', f'{tmpdirname}/model.p',
                '-M', f'{tmpdirname}/model2.p',
            ],
            capture_output=True,
            text=True
        )

        assert result2.returncode == 0, f"Second CLI command failed with error: {result2.stderr}"
        assert os.path.exists(f'{tmpdirname}/result2.parquet'), f"Second output file was not created."
        assert os.path.exists(f'{tmpdirname}/model2.p'), f"Second output model was not created."
