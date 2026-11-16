import pandas as pd


class FeatureEngineer:

    @staticmethod
    def transform(dataframe: pd.DataFrame) -> pd.DataFrame:

        df = dataframe.copy()

        # Customer engagement score
        df["customer_engagement_score"] = (
            df["website_visits"]
            + df["marketing_emails_opened"]
        )

        # Purchase history segment
        df["purchase_history_segment"] = pd.cut(
            df["previous_purchases"],
            bins=[-1, 0, 2, 4, float("inf")],
            labels=[
                "New Customer",
                "Occasional Buyer",
                "Repeat Buyer",
                "Frequent Buyer"
            ]
        )

        # Income segment
        df["income_segment"] = pd.cut(
            df["annual_income"],
            bins=[-1, 30000, 60000, float("inf")],
            labels=[
                "Low Income",
                "Medium Income",
                "High Income"
            ]
        )

        return df