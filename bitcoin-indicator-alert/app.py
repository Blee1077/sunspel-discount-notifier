import json
import requests
import boto3
import pandas as pd
import numpy as np
from redmail import EmailSender
from collections import deque

s3 = boto3.resource('s3')


def load_json(bucket: str, key: str):
    """Loads a JSON file from S3.
    
    Args:
        bucket(str): S3 bucket name
        key (str): S3 key of JSON file
        
    Returns:
        dict
    """
    content_object = s3.Object(bucket, key)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    return json.loads(file_content)
    
    
def fetch_price_data(api_key: str) -> pd.DataFrame:
    """Fetches Bitcoin price and blocked mined data.

    Args:
        api_key (str): API key for Glassnode to get blocks mined data

    Returns:
        Pandas DataFrame
    """
    # Make a get request for bitcoin price data
    resp = requests.get(
        'https://api.blockchair.com/bitcoin/blocks', 
        {'a': 'date,price(btc_usd)'},
        timeout=30
    )

    # Check if an error has occured
    resp.raise_for_status()

    # Put data into list
    date_list = []
    price_list = []
    for data in resp.json()['data']:
        date_list.append(pd.to_datetime(data['date'], format='%Y-%m-%d').replace(hour=0, minute=0, second=0, microsecond=0))
        price_list.append(data['price(btc_usd)'])

    # Convert data into pandas dataframe
    price_df = pd.DataFrame({
        'Date': pd.to_datetime(list(date_list)),
        'Price': price_list
    })

    # Filter and interpolate data
    date_filter = price_df['Date'] >= '2011-01-01'
    price_df = price_df[date_filter].reset_index(drop=True)
    price_df.loc[price_df['Price'] <= 0.01, 'Price'] = np.NAN
    price_df['Price'].interpolate(inplace=True)

    # Calculate log(x+1) for price
    price_df['PriceLog'] = np.log(price_df['Price'] + 1)
    
    # Make get request for bitcoin block count
    resp = requests.get(
        'https://api.glassnode.com/v1/metrics/blockchain/block_count',
        params={'a': 'BTC', 'api_key': api_key, 'timestamp_format': 'humanized'}
    )

    # Check if an error has occured
    resp.raise_for_status()

    # Convert data into pandas dataframe
    block_df = pd.read_json(resp.text, convert_dates=['t'])
    block_df = block_df.rename(columns={'v': 'BlocksMined', 't': 'Date'})
    block_df['Date'] = block_df['Date'].dt.tz_localize(None)

    # Merge both dataframes
    price_df = price_df.merge(block_df, how='left', on='Date')
    return price_df


def mark_top_and_bottom(df: pd.DataFrame) -> pd.DataFrame:
    then_forward_window_size = 365 * 2
    ignore_past_n_days = 365

    df['IsTop'] = 0
    df['IsBottom'] = 0

    current_forward_window_size = 365
    current_index = 0
    searching_top = True

    while True:
        window = df.loc[current_index:current_index+current_forward_window_size-1]

        if window.shape[0] == 0:
            break

        if searching_top:
            new_index = window['Price'].idxmax()
            if new_index == current_index:
                df.loc[current_index, 'IsTop'] = 1
                current_forward_window_size = then_forward_window_size
                current_index += 1
                searching_top = False
                continue
        else:
            new_index = window['Price'].idxmin()
            if new_index == current_index:
                df.loc[current_index, 'IsBottom'] = 1
                current_forward_window_size = then_forward_window_size
                current_index += 1
                searching_top = True
                continue

        current_index = new_index

    df.loc[df.shape[0]-ignore_past_n_days:, 'IsTop'] = 0
    df.loc[df.shape[0]-ignore_past_n_days:, 'IsBottom'] = 0
    return df


def fetch_block_data(df: pd.DataFrame) -> pd.DataFrame:
    blocks_per_day = 6 * 24
    halving_block = 210000

    df['IsHalving'] = 0
    df['Cycle'] = 0
    df['CyclePow'] = 0
    df['CoinIssuance'] = 0
    df['MarketCap'] = 0

    total_market_cap = 0
    current_block = halving_block
    current_cycle = 1
    current_issuance = 50

    while True:
        r = requests.get(f'https://api.blockchair.com/bitcoin/dashboards/block/{current_block}', timeout=30)
        r.raise_for_status()

        json = r.json()
        current_daily_issuance = blocks_per_day * current_issuance

        if str(current_block) not in json['data']:
            df.loc[df['Cycle'] == 0, 'Cycle'] = current_cycle
            df.loc[df['CoinIssuance'] == 0, 'CoinIssuance'] = current_daily_issuance

            first_empty_market_cap_index = df.loc[df['MarketCap'] == 0].index[0]
            last_empty_market_cap_index = df.loc[df['MarketCap'] == 0].index[-1]
            num = last_empty_market_cap_index - first_empty_market_cap_index + 1
            current_market_cap_increase = (num - 1) * current_daily_issuance

            df.loc[first_empty_market_cap_index:last_empty_market_cap_index, 'MarketCap'] = \
                np.linspace(total_market_cap, total_market_cap + current_market_cap_increase, num)
            break

        current_halving_block_market_cap = halving_block * current_issuance
        current_halving_block_date = json['data'][str(current_block)]['block']['date']
        current_halving_block_row = df[df['Date'] == current_halving_block_date]

        if current_halving_block_row.shape[0] == 1:
            current_halving_block_index = current_halving_block_row.index[0]

            df.loc[current_halving_block_index, 'IsHalving'] = 1
            df.loc[(df['Date'] < current_halving_block_date) & (df['Cycle'] == 0), 'Cycle'] = current_cycle
            df.loc[(df['Date'] < current_halving_block_date) & (df['CoinIssuance'] == 0), 'CoinIssuance'] = current_daily_issuance

            first_empty_market_cap_index = df.loc[df['MarketCap'] == 0].index[0]

            num = current_halving_block_index - first_empty_market_cap_index
            start_offset = 0

            if first_empty_market_cap_index == 0:
                expected_num = halving_block / blocks_per_day
                start_offset = (expected_num - num) * current_daily_issuance

            df.loc[first_empty_market_cap_index:current_halving_block_index-1, 'MarketCap'] = \
                np.linspace(total_market_cap + start_offset, total_market_cap + current_halving_block_market_cap - current_daily_issuance, num)

        total_market_cap += current_halving_block_market_cap
        current_block += halving_block
        current_cycle += 1
        current_issuance /= 2

    df['CoinsPerBlock'] = 50 / (2 ** (df['Cycle'] - 1))
    df['CyclePow'] = np.power(2, df['Cycle'])
    df['CoinIssuance'] = df['CoinsPerBlock'] * df['BlocksMined']
    df['CoinIssuanceUSD'] = df['Price'] * df['CoinIssuance']
    df['CoinIssuanceUSDLog'] = np.log(df['CoinIssuanceUSD'])
    df['MarketCapUSD'] = df['Price'] * df['MarketCap']
    df['MarketCapUSDLog'] = np.log(df['MarketCapUSD'])
    return df


def mark_days_since(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        indexes = df.loc[df[col] == 1].index
        df[f'DaysSince{col}'] = df.index.to_series().apply(lambda v: min([v-index if index <= v else np.NaN for index in indexes]))

    return df


def mark_bottom_price(df: pd.DataFrame) -> pd.DataFrame:
    df['BottomPrice'] = np.NaN

    bottom_indexes = df[df['IsBottom'] == 1].index

    for current_index, next_index in zip(bottom_indexes, bottom_indexes[1:]):
        df.loc[current_index:next_index-1, 'BottomPrice'] = df.loc[current_index, 'Price']

    df.loc[bottom_indexes[-1]:, 'BottomPrice'] = df.loc[bottom_indexes[-1], 'Price']
    return df


def impute_days_since(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        indexes = df.loc[df[col] == 1].index
        max_values = df.loc[indexes - 1, f'DaysSince{col}'][1:]
        avg_value = np.round(np.mean(max_values))
        df.loc[:indexes[0]-1, f'DaysSince{col}'] = np.linspace(avg_value - indexes[0] + 1, avg_value, indexes[0])

    return df


def mark_price_increase(df: pd.DataFrame) -> pd.DataFrame:
    df['PriceIncrease'] = (df['Price'] - df['BottomPrice']) / df['BottomPrice']
    df['PriceIncreaseLog'] = np.log(df['PriceIncrease'] + 1)

    df['PriceIncreaseCycle'] = df['PriceIncrease'] * df['CyclePow']
    df['PriceIncreaseCycleLog'] = np.log(df['PriceIncreaseCycle'] + 1)
    return df


def mark_top_percentage(df: pd.DataFrame) -> pd.DataFrame:
    df['TopDateBasedPercentage'] = np.NaN
    df['TopPriceBasedPercentage'] = np.NaN

    current_top_index = min(df[df['IsTop'] == 1].index)
    current_top_price = df.loc[current_top_index, 'Price']
    current_bottom_index = min(df[df['IsBottom'] == 1].index)
    current_bottom_price = df.loc[current_bottom_index, 'Price']

    while True:
        # decreasing value
        if current_top_index < current_bottom_index:
            df.loc[current_top_index:current_bottom_index, 'TopDateBasedPercentage'] = np.linspace(1, 0, current_bottom_index - current_top_index + 1)
            df.loc[current_top_index:current_bottom_index, 'TopPriceBasedPercentage'] = \
                df.loc[current_top_index:current_bottom_index, 'Price'].apply(lambda v: (v - current_bottom_price) / (current_top_price - current_bottom_price))

            mask = (df['IsTop'] == 1) & (df.index > current_top_index)
            if sum(mask) == 0:
                break

            current_top_index = min(df[mask].index)
            current_top_price = df.loc[current_top_index, 'Price']
        # increasing value
        else:
            df.loc[current_bottom_index:current_top_index, 'TopDateBasedPercentage'] = np.linspace(0, 1, current_top_index - current_bottom_index + 1)
            df.loc[current_bottom_index:current_top_index, 'TopPriceBasedPercentage'] = \
                df.loc[current_bottom_index:current_top_index, 'Price'].apply(lambda v: (v - current_bottom_price) / (current_top_price - current_bottom_price))

            mask = (df['IsBottom'] == 1) & (df.index > current_bottom_index)
            if sum(mask) == 0:
                break

            current_bottom_index = min(df[mask].index)
            current_bottom_price = df.loc[current_bottom_index, 'Price']

    df['TopDateBasedPercentage'] *= 100
    df['TopPriceBasedPercentage'] *= 100
    return df


def mark_2yma(df: pd.DataFrame) -> pd.DataFrame:
    df['2YMA'] = df['Price'].rolling(365 * 2).mean()
    df['2YMAx5'] = df['2YMA'] * 5
    df['2YMAPercentage'] = df.apply(lambda row: (row['Price'] - row['2YMA']) / (row['2YMAx5'] - row['2YMA']), axis=1)

    df['2YMAPercentageCycle'] = df['2YMAPercentage'] * df['CyclePow']
    return df


def mark_pi_cycle(df: pd.DataFrame) -> pd.DataFrame:
    df['111MA'] = df['Price'].rolling(111).mean()
    df['350MAx2'] = df['Price'].rolling(350).mean() * 2
    df['PiDifference'] = np.abs(df['111MA'] - df['350MAx2'])
    df['PiDifferenceLog'] = np.log(df['PiDifference'] + 1)

    df['111MALog'] = np.log(df['111MA'])
    df['350MAx2Log'] = np.log(df['350MAx2'])
    df['PiLogDifference'] = np.abs(df['111MALog'] - df['350MAx2Log'])
    df['PiLogDifferenceLog'] = np.log(df['PiLogDifference'] + 1)

    df['PiLogDifferenceCycle'] = df['PiLogDifferenceLog'] * df['CyclePow']
    return df


def mark_puell_multiple(df: pd.DataFrame) -> pd.DataFrame:
    df['365MA-CoinIssuanceUSD'] = df['CoinIssuanceUSD'].rolling(window=365).mean()
    df['PuellMultiple'] = df['CoinIssuanceUSD'] / df['365MA-CoinIssuanceUSD']
    df['PuellMultipleCycle'] = df['PuellMultiple'] * df['CyclePow']
    return df


def init_dataframe(api_key: str) -> pd.DataFrame:
    df = fetch_price_data(api_key)
    df = mark_top_and_bottom(df)
    df = fetch_block_data(df)
    df = mark_days_since(df, ['IsTop', 'IsBottom', 'IsHalving'])
    df = mark_bottom_price(df)
    df = impute_days_since(df, ['IsTop', 'IsBottom', 'IsHalving'])
    df = mark_price_increase(df)
    df = mark_top_percentage(df)
    df = mark_2yma(df)
    df = mark_pi_cycle(df)
    df = mark_puell_multiple(df)
    return df
    

def two_year_ma_mult_indicator(price, MA2Y, MA2Yx5):
    if price <= MA2Y:
        return "Current price is below 2-Year MA - good buying opportunity!"
    
    elif price >= MA2Yx5:
        return "Current price is above 2-Year MA 5x multiplier - good selling opportunity!"
    
    else:
        return None

    
def pi_cycle_top_indicator(MA111, MA350x2):
    if MA111 >= MA350x2:
        return "111MA >= 2*350MA - Bitcoin is overheated, good selling opportunity (expect crash soon)."
    
    else:
        return None

    
def puell_multiple_indicator(puell_multiple):
    if 0.5 >= puell_multiple >= 0.3:
        return (f"The Puell Multiple ({puell_multiple}) shows that a good buying opportunity is present.")
    
    elif 10 >= puell_multiple >= 4:
        return (f"The Puell Multiple ({puell_multiple}) shows that a good selling opportunity is present.")
    
    else:
        return None


def lambda_handler(event, context):
    api_secret = load_json(event['api_secret_bucket'], event['api_secret_json_key'])
    email_secret = load_json(event['email_secret_bucket'], event['email_secret_json_key'])
    email = EmailSender(
            host="smtp.office365.com",
            port=587,
            user_name=email_secret['email'],
            password=email_secret['password']
        )
        
    df = init_dataframe(api_key=api_secret['key'])
    
    body = []
    body.append(two_year_ma_mult_indicator(df['Price'].iloc[-1], df['2YMA'].iloc[-1], df['2YMAx5'].iloc[-1]))
    body.append(pi_cycle_top_indicator(df['111MA'].iloc[-1], df['350MAx2'].iloc[-1]))
    body.append(puell_multiple_indicator(df['PuellMultiple'].iloc[-2]))
    body = [string for string in body if string]
    
    if len(body) > 0:
        email.send(
            subject="Bitcoin Indicator Alert",
            sender=email_secret['email'],
            receivers=email_secret['email'],
            text=" \n".join(body),
        )
    
    else:
        print("No indicator alerts triggered")
        
    return body
