from bullpen import cv_utils


def test_cv_utils():
    assert hasattr(cv_utils, 'make_timeseries_splits')
