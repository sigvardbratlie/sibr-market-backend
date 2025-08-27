import pytest
import os
os.chdir('..')
from src.sibr_market_training import Clean
import pandas as pd

@pytest.fixture
def instance():
    """Fixture to create a Clean instance for testing."""
    return Clean(dataset='cars')


def test_clean_init(instance):
    """Test the initialization of the Clean class."""
    assert instance.dataset == 'cars', "Dataset should be 'cars'"
    assert instance.replace == False, "Default replace should be False"

def test_mk_num(instance):
    """Test the mk_num method."""
    df = pd.DataFrame({'col1': ['10 000 kr', '20000,-', '302 m2'], 'col2': ['4.5', '5,6l', '62.7 m2']})
    df = instance.mk_num(df,
                         int_cols=['col1'], type='int')
    df = instance.mk_num(df,int_cols =['col2'], type='float')
    assert df['col1'].dtype == 'Int64', "Column 'col1' should be of type Int64"
    assert df['col2'].dtype == 'Float64', "Column 'col2' should be of type Float64"
    assert df['col1'].iloc[0] == 10000, "First value in 'col1' should be 10000"
    assert df['col2'].iloc[0] == 4.5, "First value in 'col2' should be 4.5"
