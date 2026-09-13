import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class ExpectedBehaviorModel:
    """
    Learns expected facility-subsystem behavior from IT workload.

    The model estimates what cooling, HVAC, and pump power would
    normally be expected for a given IT load.

    IMPORTANT:
    This is an operational baseline, not a physical simulation
    and not proof of causality.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        self.models = {
            "cooling_kw": RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1
            ),
            "hvac_kw": RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1
            ),
            "pump_kw": RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1
            )
        }

        self.feature_columns = [
            "it_power_kw"
        ]

    def fit(self, df):
        """
        Train expected-behavior models.

        Only valid operational observations should be supplied
        for training.
        """

        data = df.copy()

        required = [
            "it_power_kw",
            "cooling_kw",
            "hvac_kw",
            "pump_kw"
        ]

        missing = [
            col for col in required
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # Keep training data operationally plausible.
        data = data[
            (data["it_power_kw"] > 100)
            & (data["pue"] >= 0.95)
            & (data["pue"] <= 1.15)
            & (data["cooling_kw"] >= 0)
            & (data["hvac_kw"] >= 0)
            & (data["pump_kw"] >= 0)
        ].copy()

        data = data.dropna(
            subset=required
        )

        if len(data) < 1000:
            raise ValueError(
                "Not enough valid observations to train "
                "expected-behavior models."
            )

        X = data[self.feature_columns]

        for target, model in self.models.items():
            y = data[target]
            model.fit(X, y)

        return self

    def predict_expected(self, df):
        """
        Predict expected subsystem power for each observation.
        """

        data = df.copy()

        X = data[self.feature_columns]

        for target, model in self.models.items():
            expected_column = f"expected_{target}"

            data[expected_column] = model.predict(X)

        return data

    def add_deviation_features(self, df):
        """
        Add relative deviation between actual and expected behavior.
        """

        data = df.copy()

        for subsystem in [
            "cooling",
            "hvac",
            "pump"
        ]:

            actual = f"{subsystem}_kw"
            expected = f"expected_{subsystem}_kw"

            deviation = (
                data[actual] - data[expected]
            )

            data[f"{subsystem}_deviation_kw"] = deviation

            # Relative deviation.
            data[f"{subsystem}_deviation_ratio"] = (
                deviation
                / data[expected].clip(lower=0.1)
            )

        return data

    def transform(self, df):
        """
        Predict expected behavior and generate deviation features.
        """

        data = self.predict_expected(df)
        data = self.add_deviation_features(data)

        return data