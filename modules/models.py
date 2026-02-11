import catboost as cb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive

from .feature_generation import get_features_df_and_targets
from .index_slicing import features__test_idx, features_targets__train_idx


class BaseModel:
    """Базовый класс модели."""

    def __init__(self):
        raise NotImplementedError

    def fit(
        self, train_data, val_data, id_col="ts_id", timestamp_col="timestamp", value_col="value"
    ):
        """Обучение модели на тренировочных и валидационных данных.

        Args:
        - train_data: DataFrame с тренировочными данными.
        - val_data: DataFrame с валидационными данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        Returns: None
        """
        raise NotImplementedError

    def predict(self, test_data, id_col="ts_id", timestamp_col="timestamp", value_col="value"):
        """Прогнозирование на тестовых данных.

        Args:
        - test_data: DataFrame с тестовыми данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.

        Returns:
        - predictions: DataFrame с предсказанными значениями со столбцами
            [id_col, timestamp_col, 'predicted_value'].
        """
        raise NotImplementedError


class StatsforecastModel(BaseModel):
    """Модель, использующая библиотеку statsforecast для прогнозирования."""

    def __init__(self, model, freq: str, horizon: int):
        """Инициализация модели.

        Args:
            - model: экземпляр модели из библиотеки statsforecast.
            - freq: частота временного ряда (например, 'H' для почасовых данных).
            - horizon: общий горизонт прогнозирования.

        """
        self.model = model
        self.freq = freq
        self.horizon = horizon

    def fit(
        self,
        train_data,
        val_data,
        id_col="sensor_id",
        timestamp_col="timestamp",
        value_col="value",
    ):
        """Обучение модели на тренировочных и валидационных данных.

        Args:
        - train_data: DataFrame с тренировочными данными.
        - val_data: DataFrame с валидационными данными.

        """
        # Объединяем тренировочные и валидационные данные
        combined_data = pd.concat([train_data, val_data])
        # Удаялем дубликаты, которые образовались после объединения
        # так как val_data начинается с history точек из конца train_data
        combined_data = combined_data.drop_duplicates(subset=[id_col, timestamp_col], keep="last")

        # Преобразуем данные в формат, необходимый для StatsForecast
        sf = StatsForecast(models=[self.model], freq=self.freq)
        self.sf = sf.fit(
            combined_data.rename(
                columns={id_col: "unique_id", timestamp_col: "ds", value_col: "y"}
            )
        )

    def predict(self, test_data, id_col="sensor_id", timestamp_col="timestamp", value_col="value"):
        """Прогнозирование на тестовых данных.

        Args:
            - test_data: DataFrame с тестовыми данными.

        Returns:
            - predictions: DataFrame с предсказанными значениями со столбцами
                [id_col, timestamp_col, 'predicted_value'].

        """
        forecasts = self.sf.predict(h=self.horizon)

        # Преобразуем прогнозы обратно в исходный формат
        pred_column = [col for col in forecasts.columns if col not in ["unique_id", "ds"]][0]
        predictions = forecasts[["unique_id", "ds", pred_column]].rename(
            columns={"unique_id": id_col, "ds": timestamp_col, pred_column: "predicted_value"}
        )

        return predictions


class CatBoostRecursive(BaseModel):
    """Модель CatBoost с рекурсивной стратегией прогнозирования."""

    def __init__(self, model_horizon: int, history: int, horizon: int, freq: str):
        """Инициализация модели.

        Args:
            - model_horizon: Горизонт прогнозирования модели.
            - history: Размер окна истории.
            - horizon: Общий горизонт прогнозирования.
            - freq: Частота временного ряда.

        """
        self.model_horizon = model_horizon
        self.history = history
        self.horizon = horizon
        self.freq = freq
        self.model = None

    def fit(
        self,
        train_data,
        val_data,
        id_col="sensor_id",
        timestamp_col="timestamp",
        value_col="value",
    ):
        """Обучение модели на тренировочных и валидационных данных.

        Args:
        - train_data: DataFrame с тренировочными данными.
        - val_data: DataFrame с валидационными данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        """
        # Формируем индексы для признаков и таргетов
        train_features_idx, train_targets_idx = features_targets__train_idx(
            id_column=train_data[id_col],
            series_length=len(train_data),
            model_horizon=self.model_horizon,
            history_size=self.history,
        )
        val_features_idx, val_targets_idx = features_targets__train_idx(
            id_column=val_data[id_col],
            series_length=len(val_data),
            model_horizon=self.model_horizon,
            history_size=self.history,
        )
        # Генерируем признаки и таргеты
        train_features, train_targets, categorical_features_idx = get_features_df_and_targets(
            train_data,
            train_features_idx,
            train_targets_idx,
            id_column=id_col,
            date_column=timestamp_col,
            target_column=value_col,
        )
        val_features, val_targets, _ = get_features_df_and_targets(
            val_data,
            val_features_idx,
            val_targets_idx,
            id_column=id_col,
            date_column=timestamp_col,
            target_column=value_col,
        )
        # Инициализируем и обучаем модель CatBoost
        cb_model = cb.CatBoostRegressor(
            loss_function="MultiRMSE",
            random_seed=42,
            verbose=100,
            early_stopping_rounds=100,
            iterations=1500,
            cat_features=categorical_features_idx,
        )
        train_dataset = cb.Pool(
            data=train_features, label=train_targets, cat_features=categorical_features_idx
        )
        eval_dataset = cb.Pool(
            data=val_features, label=val_targets, cat_features=categorical_features_idx
        )
        cb_model.fit(
            train_dataset,
            eval_set=eval_dataset,
            use_best_model=True,
            plot=False,
        )
        self.model = cb_model

    def predict(self, test_data, id_col="sensor_id", timestamp_col="timestamp", value_col="value"):
        """
        Прогнозирование на тестовых данных.

        Args:
        - test_data: DataFrame с тестовыми данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        Returns:
        - predictions: DataFrame с предсказанными значениями со столбцами
            [id_col, timestamp_col, 'predicted_value'].

        """
        # Последовательно прогнозируем значения
        steps = self.horizon // self.model_horizon

        for step in range(steps):
            print(f"Шаг прогнозирования {step + 1} из {steps}")

            test_features_idx, target_features_idx = features__test_idx(
                id_column=test_data[id_col],
                series_length=len(test_data),
                model_horizon=self.model_horizon,
                history_size=self.history + step * self.model_horizon,
            )
            test_features_idx = test_features_idx[:, step:]

            test_features, _, _ = get_features_df_and_targets(
                test_data,
                test_features_idx,
                target_features_idx,  # Таргеты не нужны
                id_column=id_col,
                date_column=timestamp_col,
                target_column=value_col,
            )

            test_preds = self.model.predict(test_features)

            # Заполняем предсказаниями соответствующие места в истории
            test_data.iloc[
                target_features_idx.flatten(),  # Берем только последние индексы таргетов
                test_data.columns.get_loc(value_col),
            ] = test_preds.reshape(-1, 1)

        # Оставляем в итоговом датафрейме только нужные строки и столбцы
        first_test_date = np.sort(np.unique(test_data[timestamp_col]))[self.history]
        test_data = test_data[test_data[timestamp_col] >= first_test_date]
        test_data = test_data.rename(columns={value_col: "predicted_value"})

        return test_data[[id_col, timestamp_col, "predicted_value"]].reset_index(drop=True)


class CatBoostDirect(BaseModel):
    """Модель CatBoost с прямой стратегией прогнозирования."""

    def __init__(self, model_horizon: int, history: int, horizon: int, freq: str):
        """Инициализация модели.

        Args:
            - model_horizon: Горизонт прогнозирования модели.
            - history: Размер окна истории.
            - horizon: Общий горизонт прогнозирования.
            - freq: Частота временного ряда.

        """
        self.model_horizon = model_horizon
        self.history = history
        self.horizon = horizon
        self.freq = freq
        self.models = []

    def fit(
        self,
        train_data,
        val_data,
        id_col="sensor_id",
        timestamp_col="timestamp",
        value_col="value",
    ):
        """Обучение модели на тренировочных и валидационных данных.

        Args:
        - train_data: DataFrame с тренировочными данными.
        - val_data: DataFrame с валидационными данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        """

        train_features_idx, train_targets_idx = features_targets__train_idx(
            id_column=train_data[id_col],
            series_length=len(train_data),
            model_horizon=self.horizon,
            history_size=self.history,
        )
        val_features_idx, val_targets_idx = features_targets__train_idx(
            id_column=val_data[id_col],
            series_length=len(val_data),
            model_horizon=self.horizon,
            history_size=self.history,
        )


        train_features, train_targets, categorical_features_idx = get_features_df_and_targets(
            train_data,
            train_features_idx,
            train_targets_idx,
            id_column=id_col,
            date_column=timestamp_col,
            target_column=value_col,
        )
        val_features, val_targets, _ = get_features_df_and_targets(
            val_data,
            val_features_idx,
            val_targets_idx,
            id_column=id_col,
            date_column=timestamp_col,
            target_column=value_col,
        )


        self.models = []
        for i in range(self.horizon//self.model_horizon):
            print(f"Обучение модели № {i + 1} из {self.horizon//self.model_horizon}")

            train_target_step = train_targets[:, self.model_horizon * (i):self.model_horizon * (i + 1)]
            val_target_step = val_targets[:, self.model_horizon * (i):self.model_horizon * (i + 1)]

            cb_model = cb.CatBoostRegressor(
                loss_function="MultiRMSE",
                random_seed=42,
                verbose=0,
                early_stopping_rounds=100,
                iterations=1500,
                cat_features=categorical_features_idx,
            )

            train_dataset = cb.Pool(
                data=train_features, label=train_target_step, cat_features=categorical_features_idx
            )
            eval_dataset = cb.Pool(
                data=val_features, label=val_target_step, cat_features=categorical_features_idx
            )

            cb_model.fit(
                train_dataset,
                eval_set=eval_dataset,
                use_best_model=True,
                plot=False,
            )
            self.models.append(cb_model)

    def predict(self, test_data, id_col="sensor_id", timestamp_col="timestamp", value_col="value"):
        """Прогнозирование на тестовых данных.

        Args:
        - test_data: DataFrame с тестовыми данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        Returns:
        - predictions: DataFrame с предсказанными значениями со столбцами
            [id_col, timestamp_col, 'predicted_value'].

        """
        test_data = test_data.copy()

        test_features_idx, target_features_idx = features__test_idx(
            id_column=test_data[id_col],
            series_length=len(test_data),
            model_horizon=self.horizon,
            history_size=self.history,
        )

        test_features, _, _ = get_features_df_and_targets(
            test_data,
            test_features_idx,
            target_features_idx,
            id_column=id_col,
            date_column=timestamp_col,
            target_column=value_col,
        )


        all_preds = []
        for i in range(self.horizon//self.model_horizon):
            preds = self.models[i].predict(test_features)
            all_preds.append(preds)

        # Транспонируем, чтобы получить (n_samples, horizon)
        all_preds = np.array(all_preds).T

        # Заполняем предсказаниями соответствующие места
        test_data.iloc[
            target_features_idx.flatten(),
            test_data.columns.get_loc(value_col),
        ] = all_preds.flatten()

        # Оставляем только горизонт прогнозирования
        first_test_date = np.sort(test_data[timestamp_col].unique())[self.history]
        test_data = test_data[test_data[timestamp_col] >= first_test_date]
        test_data = test_data.rename(columns={value_col: "predicted_value"})

        return test_data[[id_col, timestamp_col, "predicted_value"]].reset_index(drop=True)
