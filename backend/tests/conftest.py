import pytest
from fastapi.testclient import TestClient

import main


class FakeModel:
    def predict(self, dataframe):
        assert len(dataframe) == 1
        assert "Type" in dataframe.columns
        return [12345.678]


@pytest.fixture
def model_bundle():
    return {
        "model": FakeModel(),
        "metrics": {
            "model": "Random Forest Regressor (test fixture)",
            "r2_score": 0.91,
            "mae": 1200.0,
            "rmse": 1800.0,
        },
    }


@pytest.fixture
def client(monkeypatch, model_bundle):
    # API tests use a lightweight fake model. They do not need the private,
    # large trained .pkl artifact or the original Kaggle training dataset.
    monkeypatch.setattr(main, "load_model_bundle", lambda: model_bundle)
    main.app.dependency_overrides[main.get_model_bundle] = lambda: model_bundle

    with TestClient(main.app) as test_client:
        yield test_client

    main.app.dependency_overrides.clear()
