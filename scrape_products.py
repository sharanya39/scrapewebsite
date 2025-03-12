import os
import json
import requests
from urllib.parse import quote_plus
from playwright.sync_api import sync_playwright

def save_image(image_url, save_path):
    try:
        # Send a GET request to download the image
        img_data = requests.get(image_url).content
        with open(save_path, 'wb') as img_file:
            img_file.write(img_data)
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")

def clean_filename(filename):
    # Replace spaces, slashes, and other special characters with underscores
    return quote_plus(filename).replace('%20', '_').replace('%28', '_').replace('%29', '_')

def scrape_products(category_url, image_folder="product_images"):
    product_list = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(category_url, timeout=120000)

        # Wait for the product list container to load
        page.wait_for_selector("div.product-item", timeout=120000)

        # Find all products
        products = page.locator("div.product-item").all()

        for product in products:
            name = product.locator("h4").inner_text()  # Assuming product name is in <h4>
            price = product.locator("span.price").inner_text()  # Assuming price is in <span class="price">
            image_url = product.locator("img.home_product_img").get_attribute("src")  # Image selector

            # Clean the product name and use it for the image file name
            clean_name = clean_filename(name)
            image_filename = f"{clean_name}.jpg"
            image_path = os.path.join(image_folder, image_filename)

            # Remove the Unicode character for Indian Rupee (₹) from the price
            clean_price = price.replace("\u20b9", "").strip()

            # Save the image
            save_image(image_url, image_path)

            # Add product details to the list
            product_list.append({
                "name": name,
                "price": clean_price,  # Updated price without the ₹ symbol
                "image_path": image_path
            })

        browser.close()

    return product_list


def scrape_categories(base_url, categories, image_folder="product_images", data_file="product_data.json"):
    # Create a directory to save images if it doesn't exist
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    all_product_data = {}

    for category in categories:
        category_url = f"{base_url}/{category}"  # Construct the full URL for each category
        print(f"Scraping category: {category} from {category_url}")
        
        # Scrape product data from each category
        category_data = scrape_products(category_url, image_folder)
        all_product_data[category] = category_data

    # Save combined product data to a JSON file
    with open(data_file, 'w') as json_file:
        json.dump(all_product_data, json_file, indent=4)

    return all_product_data

# Base URL (Replace with the actual base URL)
base_url = "https://eswarspices.com/product_list"

# List of categories to scrape (Modify with actual category names)
categories = ["dry-fruits", "grains", "spices", "millets", "seeds", "pulses"]

# Scrape data for all categories
category_data = scrape_categories(base_url, categories)

print(f"Product data saved in 'product_data.json' and images in 'product_images' folder.")
