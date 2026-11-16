
import pandas as pd


class DataCleaner:

    @staticmethod
    def clean(dataframe: pd.DataFrame) -> tuple:
        original_rows = len(dataframe)

        # Remove duplicate rows
        cleaned = dataframe.drop_duplicates().copy()
        duplicates_removed = original_rows - len(cleaned)

        # Remove invalid target values
        if "purchased" in cleaned.columns:
            cleaned = cleaned[
                cleaned["purchased"].isin([0, 1])
            ].copy()

        invalid_targets_removed = (
            original_rows
            - duplicates_removed
            - len(cleaned)
        )

        # Fill missing numerical values with median
        numeric_columns = cleaned.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:
            if column == "purchased":
                continue

            median_value = cleaned[column].median()

            if pd.notna(median_value):
                cleaned[column] = cleaned[column].fillna(
                    median_value
                )

        # Fill missing categorical values with mode
        categorical_columns = cleaned.select_dtypes(
            include=["object", "category"]
        ).columns

        for column in categorical_columns:
            mode_values = cleaned[column].mode()

            if not mode_values.empty:
                cleaned[column] = cleaned[column].fillna(
                    mode_values.iloc[0]
                )

        report = {
            "original_rows": original_rows,
            "cleaned_rows": len(cleaned),
            "duplicates_removed": duplicates_removed,
            "invalid_targets_removed": invalid_targets_removed,
            "remaining_missing_values": int(
                cleaned.isnull().sum().sum()
            )
        }

        return cleaned, report
