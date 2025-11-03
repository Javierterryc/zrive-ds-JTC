import pandas as pd
from typing import Tuple

def _push_relevant_dataframe(
        df: pd.DataFrame, 
        min_products: int = 5
        ) -> pd.DataFrame:
    order_size = df.groupby("order_id").outcome.sum()
    min_order_size_index = order_size[order_size >= min_products].index
    return df[df.order_id.isin(min_order_size_index)]

def _format_date_columns(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    return (
        df
        .assign(created_at=lambda x: pd.to_datetime(x.created_at))
        .assign(order_date=lambda x: pd.to_datetime(x.order_date).dt.date)
    )

def _temporal_data_split(
        df: pd.DataFrame
)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function returns train, validation and test sets for a given pd.DataFrame.
    Since the data varies along time, the split is hard-coded as follows:
    - train_set: first 70% of the data
    - val_set: next 20% of the data (from 70% to 90%)
    - test_set: final 10% of the data (most recent observations)
    """
    daily_orders = df.groupby("order_date").order_id.nunique()
    cumsum_daily_orders = daily_orders.cumsum() / daily_orders.sum()

    train_val_cutoff = cumsum_daily_orders[cumsum_daily_orders <= 0.7].idxmax()
    val_test_cutoff = cumsum_daily_orders[cumsum_daily_orders <= 0.9].idxmax()
    train_df = df[df.order_date <= train_val_cutoff]
    val_df = df[
        (df.order_date > train_val_cutoff) & 
        (df.order_date <= val_test_cutoff)
        ]
    test_df = df[df.order_date > val_test_cutoff]

    return train_df, val_df, test_df

def _feature_label_split(
        df: pd.DataFrame, 
        label_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    This function takes a pd.DataFrame and returns the features (X) and the target label (y)
    """
    X = df.drop(columns=label_col)
    y = df[label_col]
    return X, y