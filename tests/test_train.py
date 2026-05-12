from src.config import TARGET_COLUMN
from src.data_prep import generate_demo_transactions
from src.train import stratified_sample


def test_stratified_sample_keeps_target_column():
    df = generate_demo_transactions(n_samples=1000, fraud_rate=0.03)

    sampled = stratified_sample(df, sample_size=200)

    assert len(sampled) == 200
    assert TARGET_COLUMN in sampled.columns
    assert set(sampled[TARGET_COLUMN].unique()) == {0, 1}
