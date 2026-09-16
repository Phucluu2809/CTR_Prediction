import csv
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT_DIR / "data" / "catalog" / "products.csv"
CATEGORY_ICONS = {
    "Laptop": "💻", "Máy tính bảng": "▣", "Điện thoại": "📱",
    "Làm đẹp": "💄", "Nước hoa": "🌸", "Nội thất": "🛋️",
    "Thực phẩm": "🛒", "Trang trí nhà": "🏠", "Đồ dùng bếp": "🍳",
    "Áo nam": "👕", "Giày nam": "👟", "Đồng hồ nam": "⌚",
    "Phụ kiện di động": "🔌",
}


def build_catalog():
    with CATALOG_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    catalog = []
    for row in rows:
        product_id = int(row["id"])
        catalog.append({
            "id": product_id,
            "name": row["name"],
            "brand": row["brand"],
            "category": row["category"],
            "description": row["description"],
            "price": int(row["price_vnd"]),
            "price_usd": float(row["price_usd"]),
            "rating": float(row["rating"]),
            "review_count": int(row["review_count"]),
            "discount": float(row["discount_percentage"]),
            "stock": int(row["stock"]),
            "image_url": row["image_url"],
            "image_path": str(ROOT_DIR / Path(row["image_path"])),
            "product_url": row["product_url"],
            "source": row["source"],
            "icon": CATEGORY_ICONS.get(row["category"], "🛍️"),
            "popularity": int(row["review_count"]) + float(row["rating"]) * 5 + int(row["stock"]) / 10,
        })
    return catalog
