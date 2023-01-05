import re
import os
import json
import requests
import boto3
import logging
import pandas as pd
from redmail import EmailSender
from bs4 import BeautifulSoup
from pretty_html_table import build_table

# Set up logging
logger = logging.getLogger('sunspel-discount-notifier')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

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


def convert_dict_to_dataframe(prod_dict: dict, convert_dict: dict):
    """Converts a dictionary into a Pandas DataFrame.
    
    Args:
        prod_dict (dict): Dictionary to convert into dataframe
        
    Returns:
        Pandas DataFrame
    """
    # Convert the dict into a dataframe
    prod_df = pd.DataFrame.from_records(prod_dict).T.reset_index()
    prod_df = prod_df.drop(columns=['index'])
    
    # Check required keys are in dict
    try:
        assert set(prod_df.columns).intersection(convert_dict.keys()) == set(convert_dict.keys())
    except AssertionError as e:
        missing_keys = set(convert_dict.keys()).difference(prod_dict.keys())
        logger.error(f"Missing following keys in scraped product dictionary: {', '.join(missing_keys)}")
        raise e

    # Cast columns
    for col, dtype in convert_dict.items():
        prod_df[col] = prod_df[col].astype(dtype)

    return prod_df


def size_availability(url: str):
    """Checks the sizes available from a product URL.
    
    Args:
        url (str): URL of the product page to check
        
    Returns:
        list of in-stock sizes
    """
    page = requests.get(url, headers = {'User-agent': 'tmp'})
    soup = BeautifulSoup(page.content, "html.parser")
    
    # Try and see if this works
    try:
        prods_json_txt = soup.find('script', attrs={'type': "application/json", 'data-variant-selects-el':'data'}).text
        size_details = json.loads(prods_json_txt)
        
    except AttributeError:
        # If not then fall back to a different approach, if that fails then return list with failed string
        try:
            prods_json_txt = soup.find('script', attrs={'type': "application/json", 'data-variant-radios-el':'data'}).text
            size_details = json.loads(prods_json_txt)
            
        except AttributeError:
            return ['Failed to get sizes']

    avail_dict = {}
    for detail in size_details:
        size = detail['title']
        avail = detail['available']

        if avail not in avail_dict:
            avail_dict[avail] = [size]
        else:
            avail_dict[avail].append(size)

    return avail_dict.get(True, ['None'])
    

def lambda_handler(event, context):
    
    # Get the minimum discount percentage for sending out an email
    MIN_DISCOUNT_PERC = int(os.environ['MIN_DISCOUNT_PERC'])

    # Define base URL and get all products from the page
    BASE_URL = "https://www.sunspel.com/"
    RIV_POLO_URL = "uk/mens/polo-shirts/riviera-polo.html"
    page = requests.get(BASE_URL+RIV_POLO_URL, headers = {'User-agent': 'tmp'})
    soup = BeautifulSoup(page.content, "html.parser")
    product_list = soup.find_all("li", class_="prd-List_Item")
    assert len(product_list) > 0, "No products found from HTML, web page has changed."
    
    # Loop through each product
    prod_dict = {}
    for prod in product_list:
        
        # Get the product type
        try:
            prod_type = prod.find("h3", class_="prd-Card_Title").get_text().strip()
        except KeyError as e:
            logger.error("Unable to get product type from product HTML, web page has changed.")
            raise e
        
        # If it isn't the one I want then move on
        if len(prod_type.split(' ')) != 3:
            continue
        
        # Get product ID, colour, URL, image URL, and price
        try:
            prod_id = int(prod['id'].split('-')[-1])
        except KeyError as e:
            logger.error("Unable to get product ID from product HTML, web page has changed.")
            raise e

        try:
            prod_colour = prod.find("p", class_="prd-Card_Colour").get_text().strip()
        except KeyError as e:
            logger.error("Unable to get product colour from product HTML, web page has changed.")
            raise e

        try:
            prod_url = BASE_URL[:-1] + prod.find("a", class_="util-FauxLink_Link")['href']
            if '.html' not in prod_url:
                prod_url = prod_url + '.html'
        except Exception as e:
            logger.error("Unable to get product URL from product HTML, web page has changed.")
            raise e

        try:
            prod_img_url = (
                'https:' + (
                    prod
                    .find('div', class_='prd-Card_Image')
                    .find('div', class_='rsp-Image')
                    .find('img', class_='rsp-Image_Image')['src']
                    )
                ).split('?')[0]
        except Exception as e:
            logger.error("Unable to get product image URL from product HTML, web page has changed.")
            raise e

        try:
            # Attempt to get Sale price
            sale_price_lst = prod.find('p', class_='prd-Card_Price').find('span', class_='prd-Card_SalePrice').get_text().split('\n')
            prod_price = float(
                [sale_price_str.strip() for sale_price_str in sale_price_lst
                    if len(sale_price_str.strip()) > 0 and '£' in sale_price_str][0]
                .replace('£', '')
            )
        except AttributeError as e:
            logger.info("Couldn't get sale price, attempting to get regular price.")
            try:
                prod_price = float(prod.find('p', class_='prd-Card_Price').get_text().replace('£', ''))
            except Exception as e:
                logger.error("Unable to get product price from product HTML, web page has changed.")
                raise e
        except Exception as e:
            logger.error("Unable to get product price from product HTML, web page has changed.")
            raise e
        
        # Store in dictionary
        prod_dict[prod_id] = {}
        prod_dict[prod_id]['Link URL'] = prod_url
        prod_dict[prod_id]['Image'] = prod_img_url
        prod_dict[prod_id]['Price (£)'] = prod_price
        prod_dict[prod_id]['Colour'] = prod_colour
    
    # Convert product dictionary into dataframe
    convert_dict = {
        'Link URL': str,
        'Price (£)': float,
        'Colour': str
    }
    df = convert_dict_to_dataframe(prod_dict, convert_dict)
    
    # Calculate discount
    df['Discount (%)'] = (100 * (1- (df['Price (£)'] / df['Price (£)'].max()))).round(2)
    
    # Get sizes in-stock for each product
    df['Available Sizes'] = df['Link URL'].apply(lambda x: ', '.join(size_availability(x)))
    
    # Calculate how many products are above the minimum discount threshold
    abv_disc_thresh = (df['Discount (%)'] >= MIN_DISCOUNT_PERC).sum()
    
    # Only if there's products above threshold then send an email
    if abv_disc_thresh > 0:
        html = """
        <html>
          <head></head>
          <body>
            Only showing polo shirts with a discount of {0}% or more!
            {1}
          </body>
        </html>
        """.format(
            MIN_DISCOUNT_PERC,
            build_table(
                df[df['Discount (%)']>=MIN_DISCOUNT_PERC].sort_values(by='Discount (%)', ascending=False),
                'blue_light'
            )
        )
        
        # Make links clickable in the html
        urls = re.findall(r'(?=https://).*.html', html)
        for url in urls:
            html = html.replace(url, f'<a href={url}>{url}</a>')
            
        # Put thumbnails into the html
        img_urls = re.findall(r'(?=https://).*.jpg', html)
        for img_url in img_urls:
            html = html.replace(img_url, f'<img src={img_url} alt="" width="135" height="180">')
        
        # Set up email
        email_secret = load_json(os.environ['EMAIL_SECRET_BUCKET'], os.environ['EMAIL_SECRET_JSON_KEY'])
        email = EmailSender(
                host=email_secret['host'],
                port=email_secret['port'],
                username=email_secret['sender_email'],
                password=email_secret['password']
            )
    
        # Send email
        email.send(
            subject=f"Sunspel has discounts up to {int(df['Discount (%)'].max())}% on Riviera polo shirts!",
            sender=email_secret['sender_email'],
            receivers=[email_secret['receiver_email']],
            html=html
        )
        
    return True
