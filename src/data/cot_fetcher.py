import pandas as pd
from cftc_cot import cot_download_year
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def fetch_cot_gold(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetches Gold COT data from CFTC using cftc-cot library.
    Commodity code: 088691 (COMEX Gold)
    """
    all_data = []

    for year in range(start_year, end_year + 1):
        try:
            # cot_download_year returns a DataFrame for the given year and report type
            df_year = cot_download_year(year=year, cot_report_type='legacy')

            # Filter for Gold futures
            gold_df = df_year[df_year['CFTC_Commodity_Code'].astype(str) == '088691'].copy()

            if not gold_df.empty:
                # Legacy columns
                gold_df['week_date'] = pd.to_datetime(gold_df['Report_Date_as_YYYY_MM_DD']).dt.date
                gold_df['net_long'] = gold_df['NonCommercial_Positions_Long_All'].astype(int) - \
                                    gold_df['NonCommercial_Positions_Short_All'].astype(int)
                gold_df['net_short'] = gold_df['Commercial_Positions_Short_All'].astype(int) - \
                                     gold_df['Commercial_Positions_Long_All'].astype(int)
                gold_df['noncomm_net'] = gold_df['NonCommercial_Positions_Long_All'].astype(int) - \
                                       gold_df['NonCommercial_Positions_Short_All'].astype(int)
                gold_df['comm_net'] = gold_df['Commercial_Positions_Long_All'].astype(int) - \
                                    gold_df['Commercial_Positions_Short_All'].astype(int)

                all_data.append(gold_df[['week_date', 'net_long', 'net_short', 'noncomm_net', 'comm_net']])
        except Exception as e:
            logger.error(f"Error fetching COT for year {year}: {e}")

    if not all_data:
        return pd.DataFrame(columns=['week_date', 'net_long', 'net_short', 'noncomm_net', 'comm_net'])

    result_df = pd.concat(all_data).sort_values('week_date').drop_duplicates('week_date')
    return result_df.reset_index(drop=True)

def validate_cot(df: pd.DataFrame) -> bool:
    """
    Validates COT data quality.
    """
    if df.empty:
        logger.warning("Validation failed: DataFrame is empty")
        return False

    # Check no NaN
    if df[['net_long', 'net_short', 'noncomm_net', 'comm_net']].isna().any().any():
        logger.error("Validation failed: NaNs found in numeric columns")
        return False

    # Check gaps (Warning only, do not block storage if CFTC is delayed)
    dates = pd.to_datetime(df['week_date']).sort_values()
    diffs = dates.diff().dropna()
    if (diffs > pd.Timedelta(days=14)).any():
        max_gap = diffs.max()
        logger.warning(f"Validation warning: Date gap found ({max_gap}). CFTC report might be delayed.")

    # Check net_long range: -300000 to +300000
    if (df['net_long'] < -300000).any() or (df['net_long'] > 300000).any():
        logger.error("Validation failed: net_long outside range [-300000, 300000]")
        return False

    return True

def fetch_and_store_cot(start_year: int, end_year: int, db_conn):
    """
    Calls fetch_cot_gold() and upserts into cot_xauusd table.
    """
    df = fetch_cot_gold(start_year, end_year)

    if df.empty:
        logger.warning(f"No COT data fetched for range {start_year}-{end_year}")
        return

    # Validate before storage
    is_valid = validate_cot(df)
    if not is_valid:
        logger.error("COT data validation failed. Skipping storage.")
        return

    rows_inserted = 0
    rows_updated = 0

    with db_conn.cursor() as cur:
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO cot_xauusd (week_date, net_long, net_short, noncomm_net, comm_net)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (week_date) DO UPDATE SET
                    net_long = EXCLUDED.net_long,
                    net_short = EXCLUDED.net_short,
                    noncomm_net = EXCLUDED.noncomm_net,
                    comm_net = EXCLUDED.comm_net
                RETURNING (xmax = 0) AS inserted;
            """, (row['week_date'], row['net_long'], row['net_short'], row['noncomm_net'], row['comm_net']))

            inserted = cur.fetchone()[0]
            if inserted:
                rows_inserted += 1
            else:
                rows_updated += 1

        db_conn.commit()

    logger.info(f"COT Sync Complete: {rows_inserted} inserted, {rows_updated} updated.")
