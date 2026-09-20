import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.catalog import build_catalog
from src.inference import CTRPredictor, load_demo_pool

ROOT_DIR = Path(__file__).resolve().parent
METRICS_PATH = ROOT_DIR / "outputs" / "results" / "deepfm_metrics.json"
CATALOG = build_catalog()
PAGE_SIZE = 12
SESSION_MAX_UPLIFT = 0.50
CATALOG_PRIOR_STRENGTH = 1.50

st.set_page_config(page_title="CTR Shop — DeepFM", page_icon="🛍️", layout="wide")
st.markdown("""
<style>
.stApp{background:#f6f7fb}.block-container{padding-top:1.2rem;padding-bottom:3rem}
.hero{background:linear-gradient(120deg,#111827,#3730a3);border-radius:22px;padding:26px 32px;color:white;margin-bottom:18px;box-shadow:0 14px 35px #312e8125}
.hero h1{margin:0;font-size:2.15rem}.hero p{margin:7px 0 0;color:#dbeafe}
.product{background:white;border:1px solid #e7e9f2;border-radius:17px;padding:17px;min-height:205px;box-shadow:0 7px 22px #0f172a0d}
.meta{color:#6b7280;font-size:.79rem}
.pname{font-weight:750;font-size:1rem;color:#111827;min-height:47px;margin:5px 0}
.price{color:#e11d48;font-weight:800;font-size:1.08rem}.old{color:#9ca3af;text-decoration:line-through;font-size:.78rem}
.badge{display:inline-block;background:#eef2ff;color:#4338ca;border-radius:999px;padding:5px 9px;font-weight:700;font-size:.78rem;margin-top:9px}
.notice{background:#eff6ff;border:1px solid #bfdbfe;color:#1e40af;border-radius:12px;padding:11px 14px;margin:8px 0 18px}
.active{background:#ecfdf5;border-color:#a7f3d0;color:#047857}
div[data-testid="stMetric"]{background:white;border:1px solid #e7e9f2;border-radius:14px;padding:12px}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def predictor():
    return CTRPredictor()


@st.cache_data
def demo_pool():
    return load_demo_pool(5000)


@st.cache_data
def metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def money(value):
    return f"{value:,.0f} ₫".replace(",", ".")


def stable_seed(values):
    raw = "|".join(map(str, values))
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12], 16)


def minmax(values):
    values = values.astype(float)
    value_range = values.max() - values.min()
    if value_range == 0:
        return pd.Series(0.5, index=values.index)
    return (values - values.min()) / value_range


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -20, 20)))


def score_catalog(context, click_history):
    pool = demo_pool()
    seed = stable_seed(context)
    indices = [(seed + item["id"] * 397) % len(pool) for item in CATALOG]
    outputs = predictor().predict(pool.iloc[indices].reset_index(drop=True))
    frame = pd.DataFrame(CATALOG)

    # Criteo rows are anonymous and cannot be mapped one-to-one to catalog products.
    # DeepFM therefore provides a context-level prior, while actual catalog metadata
    # differentiates products during cold start.
    deepfm_prior = float(np.mean(outputs["ctr"]))
    deepfm_prior = float(np.clip(deepfm_prior, 1e-6, 1 - 1e-6))
    frame["deepfm_prior"] = deepfm_prior

    rating_score = frame["rating"].clip(0, 5) / 5
    popularity_score = minmax(frame["popularity"])
    discount_score = minmax(frame["discount"])
    frame["catalog_prior"] = (
        0.55 * rating_score
        + 0.30 * popularity_score
        + 0.15 * discount_score
    )

    prior_logit = np.log(deepfm_prior / (1 - deepfm_prior))
    centered_catalog_prior = frame["catalog_prior"] - frame["catalog_prior"].mean()
    frame["base_ctr"] = sigmoid(
        prior_logit + CATALOG_PRIOR_STRENGTH * centered_catalog_prior
    )
    frame["session_score"] = 0.0

    if not click_history:
        frame["ctr"] = frame["base_ctr"]
        return frame.sort_values(["ctr", "popularity"], ascending=False).reset_index(drop=True)

    catalog_by_id = {item["id"]: item for item in CATALOG}
    clicked_items = [catalog_by_id[product_id] for product_id in click_history if product_id in catalog_by_id]
    if not clicked_items:
        frame["ctr"] = frame["base_ctr"]
        return frame.sort_values(["ctr", "popularity"], ascending=False).reset_index(drop=True)

    clicked = pd.DataFrame(clicked_items)
    category_counts = clicked["category"].value_counts().to_dict()
    brand_counts = clicked["brand"].value_counts().to_dict()

    # Count-based interests never become weaker merely because another item is clicked.
    # The exponential saturation keeps repeated clicks from dominating the base model.
    category_interest = 1 - np.exp(-frame["category"].map(category_counts).fillna(0))
    brand_interest = 1 - np.exp(-frame["brand"].map(brand_counts).fillna(0))

    # Use the best match against any clicked price instead of a moving median. Adding a
    # click can therefore improve, but never reduce, an item's price similarity.
    clicked_prices = clicked["price"].drop_duplicates().to_numpy(dtype=float)
    product_prices = frame["price"].to_numpy(dtype=float)[:, None]
    price_similarity = np.exp(
        -np.abs(product_prices - clicked_prices[None, :])
        / np.maximum(clicked_prices[None, :], 1)
    ).max(axis=1)

    # Session behavior is a separate, bounded signal. Category interest gates the
    # brand/price signals so a Laptop click does not boost unrelated categories.
    frame["session_score"] = category_interest * (
        0.70 + 0.20 * brand_interest + 0.10 * price_similarity
    )

    # Preserve the DeepFM probability and add only a bounded uplift for ranking.
    # This stays in [base_ctr, 1] and is monotonic as matching clicks accumulate.
    frame["ctr"] = frame["base_ctr"] + (
        1 - frame["base_ctr"]
    ) * SESSION_MAX_UPLIFT * frame["session_score"]
    return frame.sort_values(["ctr", "popularity"], ascending=False).reset_index(drop=True)


def register_click(product_id):
    st.session_state.click_history.append(product_id)
    st.rerun()


if "click_history" not in st.session_state:
    st.session_state.click_history = []

with st.sidebar:
    st.title("🛍️ CTR Shop")
    st.caption("Kho sản phẩm cá nhân hóa")
    st.divider()
    st.subheader("Khách hàng")
    age = st.selectbox("Nhóm tuổi", ["18–24", "25–34", "35–44", "45+"])
    gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])
    region = st.selectbox("Khu vực", ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Khác"])
    st.subheader("Bối cảnh")
    device = st.selectbox("Thiết bị", ["Mobile", "Desktop", "Tablet"])
    period = st.selectbox("Thời gian", ["Buổi sáng", "Buổi chiều", "Buổi tối", "Đêm khuya"])
    day = st.selectbox("Ngày", ["Ngày thường", "Cuối tuần"])
    browser = st.selectbox("Trình duyệt", ["Chrome", "Safari", "Edge", "Firefox"])
    st.divider()
    st.subheader(f"Lịch sử click ({len(st.session_state.click_history)})")
    if st.session_state.click_history:
        for product_id in st.session_state.click_history[-5:][::-1]:
            product = next(item for item in CATALOG if item["id"] == product_id)
            st.caption(f"{product['icon']} {product['name']}")
        if st.button("Xóa lịch sử", width="stretch"):
            st.session_state.click_history = []
            st.rerun()
    else:
        st.caption("Chưa có tương tác")

context = [age, gender, region, device, period, day, browser]
ranking = score_catalog(context, st.session_state.click_history)
metric = metrics()

st.markdown("""<div class="hero"><h1>Khám phá sản phẩm dành cho bạn</h1><p>100 sản phẩm được xếp hạng bằng DeepFM kết hợp hành vi trong phiên.</p></div>""", unsafe_allow_html=True)
metric_cols = st.columns(4)
metric_cols[0].metric("Sản phẩm", "100")
metric_cols[1].metric("Xếp hạng", "Hybrid DeepFM")
metric_cols[2].metric("DeepFM Test AUC", f"{metric['test_auc']:.4f}")
metric_cols[3].metric("Lượt tương tác", len(st.session_state.click_history))

if st.session_state.click_history:
    st.markdown("<div class='notice active'>Điểm hybrid đang kết hợp CTR DeepFM với mức nâng có giới hạn cho các danh mục đã click.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='notice'>Cold-start: CTR nền DeepFM được kết hợp với đánh giá, độ phổ biến và ưu đãi của catalog. Hãy bấm Xem sản phẩm để bắt đầu cá nhân hóa.</div>", unsafe_allow_html=True)

filters = st.columns([2, 1, 1])
search = filters[0].text_input("Tìm sản phẩm", placeholder="Nhập tên hoặc thương hiệu...")
category = filters[1].selectbox(
    "Danh mục",
    ["Tất cả"] + sorted(ranking["category"].unique().tolist()),
    help="Đây chỉ là bộ lọc hiển thị; nút Xem sản phẩm mới ghi nhận sở thích.",
)
sort_mode = filters[2].selectbox("Sắp xếp", ["CTR cao nhất", "Giá thấp nhất", "Giá cao nhất"])

visible = ranking.copy()
if search:
    mask = visible["name"].str.contains(search, case=False, na=False) | visible["brand"].str.contains(search, case=False, na=False)
    visible = visible[mask]
if category != "Tất cả":
    visible = visible[visible["category"] == category]
if sort_mode == "Giá thấp nhất":
    visible = visible.sort_values("price")
elif sort_mode == "Giá cao nhất":
    visible = visible.sort_values("price", ascending=False)

page_count = max(1, math.ceil(len(visible) / PAGE_SIZE))
page = st.selectbox("Trang", range(1, page_count + 1), format_func=lambda value: f"{value}/{page_count}")
page_frame = visible.iloc[(page - 1) * PAGE_SIZE:page * PAGE_SIZE]
st.caption(f"Hiển thị {len(page_frame)} trong tổng số {len(visible)} sản phẩm")

for row_start in range(0, len(page_frame), 4):
    columns = st.columns(4)
    product_row = page_frame.iloc[row_start:row_start + 4]
    for column, (_, item) in zip(columns, product_row.iterrows()):
        with column:
            st.image(item["image_path"], width="stretch")
            st.markdown(f"""
            <div class="product">
                <div class="meta">{item['category']} · {item['brand']} · -{item['discount']:.0f}%</div>
                <div class="pname">{item['name']}</div>
                <div class="price">{money(item['price'])}</div>
                <div class="meta">★ {item['rating']:.1f}/5 · Còn {item['stock']} sản phẩm</div>
                <span class="badge">Điểm CTR {item['ctr']:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Xem sản phẩm", key=f"product_{int(item['id'])}", width="stretch"):
                register_click(int(item["id"]))

st.divider()
if visible.empty:
    st.info("Không có sản phẩm phù hợp với bộ lọc hiện tại.")
else:
    # Charts and details must use the same filtered product set as the cards above.
    chart_ranking = visible.sort_values(
        ["ctr", "popularity"], ascending=False
    ).reset_index(drop=True)
    chart_left, chart_right = st.columns([1.35, 1])
    with chart_left:
        st.subheader("Top sản phẩm trong danh sách đang lọc")
        st.bar_chart(
            chart_ranking.head(10).set_index("name")[["ctr"]],
            horizontal=True,
            color="#4f46e5",
            height=390,
        )
    with chart_right:
        st.subheader("Chi tiết xếp hạng")
        top_product_ids = chart_ranking.head(20)["id"].astype(int).tolist()
        product_names = chart_ranking.set_index("id")["name"].to_dict()
        selected_id = st.selectbox(
            "Sản phẩm",
            top_product_ids,
            format_func=lambda value: product_names[value],
        )
        selected = chart_ranking[chart_ranking["id"] == selected_id].iloc[0]
        detail_cols = st.columns(2)
        detail_cols[0].metric("Điểm CTR hybrid", f"{selected['ctr']:.2%}")
        detail_cols[1].metric("CTR cold-start", f"{selected['base_ctr']:.2%}")
        signals = pd.DataFrame(
            {
                "Giá trị": [
                    selected["deepfm_prior"],
                    selected["catalog_prior"],
                    selected["session_score"],
                ]
            },
            index=["DeepFM prior", "Catalog prior", "Session interest"],
        )
        st.bar_chart(signals, color="#10b981", height=245)

st.caption("Lịch sử tương tác chỉ được tạo sau khi khách hàng bấm xem sản phẩm và được giữ trong phiên hiện tại.")
