import re
import os
import json
import requests
import boto3
import pandas as pd
from redmail import EmailSender
from bs4 import BeautifulSoup
from pretty_html_table import build_table

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


def convert_dict_to_dataframe(prod_dict: dict):
    """Converts a dictionary into a Pandas DataFrame.
    
    Args:
        prod_dict (dict): Dictionary to convert into dataframe
        
    Returns:
        Pandas DataFrame
    """
   # Specify the dtypes of columns to cast
    convert_dict = {
        'url': str,
        'price': float,
        'colour': str
    }

    # Convert the dict into a dataframe
    prod_df = pd.DataFrame.from_records(prod_dict).T.reset_index()
    prod_df = prod_df.drop(columns=['index'])

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
    
    avail_dict = {}
    size_details = soup.find_all('span', class_="nosto_sku")
    for detail in size_details:
        size = detail.find('span', class_="size").get_text()
        avail = detail.find('span', class_="availability").get_text()

        if avail not in avail_dict:
            avail_dict[avail] = [size]
        else:
            avail_dict[avail].append(size)

    return avail_dict.get('InStock', ['None'])
    

def lambda_handler(event, context):
    
    # Get the minimum discount percentage for sending out an email
    MIN_DISCOUNT_PERC = int(os.environ['MIN_DISCOUNT_PERC'])

    # Define base URL and get all products from the page
    URL = "https://www.sunspel.com/uk/mens/polo-shirts/riviera-polo.html"
    page = requests.get(URL, headers = {'User-agent': 'tmp'})
    soup = BeautifulSoup(page.content, "html.parser")
    product_list = soup.find_all("article", class_="product-item-info")

    # Loop through each product
    prod_dict = {}
    for prod in product_list:
        
        # Get the product type
        prod_type = prod.find("a", class_="product-item-link").get_text().strip().replace(r'\n', '')
        
        # If it isn't the one I want then move on
        if len(prod_type.split(' ')) != 3:
            continue
        
        # Get product ID, colour, URL, image URL, and price
        prod_id = int(prod.find("div", class_="price-box price-final_price")['data-product-id'])
        prod_colour = prod.find("div", class_="product-item-colour").get_text().split('\n')[1].strip()
        prod_url = prod.find("a", class_="product-item-link")['href']
        prod_img_url = prod.find('img', alt=prod_type)['data-src']
        
        for price in prod.find_all("span", class_="price-wrapper"):
            if price['data-price-type']=="finalPrice":
                prod_price = float(price['data-price-amount'])
        
        # Store in dictionary
        prod_dict[prod_id] = {}
        prod_dict[prod_id]['url'] = prod_url
        prod_dict[prod_id]['img_url'] = prod_img_url
        prod_dict[prod_id]['price'] = prod_price
        prod_dict[prod_id]['colour'] = prod_colour
    
    # Convert product dictionary into dataframe
    df = convert_dict_to_dataframe(prod_dict)
    
    # Calculate discount
    df['discount (%)'] = (100 * (1- (df['price'] / df['price'].max()))).round(2)
    
    # Get sizes in-stock for each product
    df['sizes available'] = df['url'].apply(lambda x: ', '.join(size_availability(x)))
    
    # Calculate how many products are above the minimum discount threshold
    abv_disc_thresh = (df['discount (%)'] >= MIN_DISCOUNT_PERC).sum()
    
    # Only if there's products above threshold then send an email
    if abv_disc_thresh > 0:
        html = """\
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
                df[df['discount (%)']>=MIN_DISCOUNT_PERC].sort_values(by='discount (%)', ascending=False),
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
            subject=f"Sunspel has discounts up to {int(df['discount (%)'].max())}% on Riviera polo shirts!",
            sender=email_secret['sender_email'],
            receivers=[email_secret['receiver_email']],
            html=html
        )
        
    return True
