import csv
import time
from pathlib import Path

import requests


SOURCE_URL = "https://dummyjson.com/products?limit=100"
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "catalog"
IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_CSV = OUTPUT_DIR / "products.csv"
PRODUCT_LIMIT = 100
USD_TO_VND = 25_000

CATEGORY_NAMES = {
    "beauty": "Làm đẹp", "fragrances": "Nước hoa", "furniture": "Nội thất",
    "groceries": "Thực phẩm", "home-decoration": "Trang trí nhà",
    "kitchen-accessories": "Đồ dùng bếp", "laptops": "Laptop",
    "mens-shirts": "Áo nam", "mens-shoes": "Giày nam", "mens-watches": "Đồng hồ nam",
    "mobile-accessories": "Phụ kiện di động", "motorcycle": "Xe máy",
    "skin-care": "Chăm sóc da", "smartphones": "Điện thoại",
    "sports-accessories": "Phụ kiện thể thao", "sunglasses": "Kính mắt",
    "tablets": "Máy tính bảng", "tops": "Thời trang",
    "vehicle": "Ô tô", "womens-bags": "Túi nữ", "womens-dresses": "Váy nữ",
    "womens-jewellery": "Trang sức nữ", "womens-shoes": "Giày nữ",
    "womens-watches": "Đồng hồ nữ",
}


def download_image(session, image_url, product_id):
    suffix = Path(image_url.split("?", 1)[0]).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".webp"
    relative_path = Path("data") / "catalog" / "images" / f"product_{product_id:03d}{suffix}"
    destination = ROOT_DIR / relative_path
    response = session.get(image_url, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return relative_path.as_posix()


def transform_product(raw, local_id, session):
    price_usd = float(raw["price"])
    reviews = raw.get("reviews") or []
    image_url = raw.get("thumbnail") or raw.get("images", [""])[0]
    return {
        "id": local_id,
        "source_id": raw["id"],
        "name": raw["title"],
        "brand": raw.get("brand") or "Khác",
        "category": CATEGORY_NAMES.get(raw["category"], raw["category"].replace("-", " ").title()),
        "category_slug": raw["category"],
        "description": raw.get("description", ""),
        "price_usd": price_usd,
        "price_vnd": int(round(price_usd * USD_TO_VND, -4)),
        "discount_percentage": float(raw.get("discountPercentage", 0)),
        "rating": float(raw.get("rating", 0)),
        "review_count": len(reviews),
        "stock": int(raw.get("stock", 0)),
        "sku": raw.get("sku", ""),
        "availability": raw.get("availabilityStatus", ""),
        "product_url": f"https://dummyjson.com/products/{raw['id']}",
        "image_url": image_url,
        "image_path": download_image(session, image_url, local_id),
        "source": "DummyJSON Products API",
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for old_image in IMAGE_DIR.glob("product_*.*"):
        old_image.unlink()
    session = requests.Session()
    session.headers.update({"User-Agent": "CTRPredictionStudentProject/1.0"})

    response = session.get(SOURCE_URL, timeout=30)
    response.raise_for_status()
    raw_products = response.json().get("products", [])[:PRODUCT_LIMIT]
    if len(raw_products) != PRODUCT_LIMIT:
        raise RuntimeError(f"Expected {PRODUCT_LIMIT} products, received {len(raw_products)}.")

    products = []
    for local_id, raw in enumerate(raw_products, start=1):
        products.append(transform_product(raw, local_id, session))
        time.sleep(0.03)

    fieldnames = list(products[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(products)

    print(f"Saved {len(products)} products to {OUTPUT_CSV}")
    print(f"Downloaded {len(products)} images to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
