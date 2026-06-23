from sqlalchemy import create_engine
import pandas as pd

DATABASE_URL = "sqlite:///data/vol_forecaster.db"

def get_engine():
    return create_engine(DATABASE_URL)


def save_to_db(df: pd.DataFrame, table_name: str):
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists='replace', index=True)
    print(f"Saved {len(df)} rows to table '{table_name}'")


def load_from_db(table_name: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='date', parse_dates=['date'])