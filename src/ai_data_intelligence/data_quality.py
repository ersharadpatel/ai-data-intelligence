
import pandas as pd


class DataQualityAnalyzer:

    @staticmethod
    def analyze(dataframe: pd.DataFrame) -> dict:
        total_rows, total_columns = dataframe.shape
        total_cells = total_rows * total_columns

        # Missing value analysis
        missing_by_column = dataframe.isnull().sum()

        missing_report = {
            column: int(count)
            for column, count in missing_by_column.items()
            if count > 0
        }

        # Duplicate analysis
        duplicate_rows = int(dataframe.duplicated().sum())

        # Data type analysis
        data_types = {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        }

        # Target value validation
        target_report = {}

        for column in dataframe.columns:
            if str(column).lower() in {"purchased", "target", "label"}:
                invalid_mask = (
                    ~dataframe[column].isin([0, 1])
                    & dataframe[column].notna()
                )

                invalid_values = (
                    dataframe.loc[invalid_mask, column]
                    .unique()
                    .tolist()
                )

                target_report[column] = {
                    "valid_values": [0, 1],
                    "invalid_values": [
                        str(value) for value in invalid_values
                    ],
                    "invalid_count": int(invalid_mask.sum())
                }

        # Constant column detection
        constant_columns = [
            column
            for column in dataframe.columns
            if dataframe[column].nunique(dropna=False) <= 1
        ]

        # Numerical outlier detection using IQR
        outlier_report = {}

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:
            column_name = str(column).lower()

            # Skip identifier and target columns
            if (
                column_name == "id"
                or column_name.endswith("_id")
                or column_name in {"purchased", "target", "label"}
                or dataframe[column].nunique() <= 2
            ):
                continue

            series = dataframe[column].dropna()

            if len(series) < 4:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outliers = series[
                (series < lower_bound) |
                (series > upper_bound)
            ]

            outlier_report[column] = {
                "count": int(len(outliers)),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2)
            }

        # Calculate quality scores
        missing_total = int(missing_by_column.sum())

        completeness = (
            ((total_cells - missing_total) / total_cells) * 100
            if total_cells else 0
        )

        uniqueness = (
            ((total_rows - duplicate_rows) / total_rows) * 100
            if total_rows else 0
        )

        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "missing_values": missing_total,
            "missing_by_column": missing_report,
            "duplicate_rows": duplicate_rows,
            "data_types": data_types,
            "target_validation": target_report,
            "constant_columns": constant_columns,
            "outliers": outlier_report,
            "completeness_score": round(completeness, 2),
            "uniqueness_score": round(uniqueness, 2)
        }
