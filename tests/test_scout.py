import yaml
import polars as pl
from xirescore.XiRescore import XiRescore

def test_scout():
    with open('tests/fixtures/config_scout.yaml', 'r') as file:
        options = yaml.safe_load(file)

    rescorer = XiRescore(
        input_path='tests/fixtures/scout.csv',
        options=options,
    )
    rescorer.run()
    df_rescored = rescorer.get_rescored_output()
    pass
