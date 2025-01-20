from bullpen import model_utils


def test_model_utils():
    assert model_utils.MODEL_DIR.exists()
