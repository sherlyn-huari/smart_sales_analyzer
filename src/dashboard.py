"""Streamlit dashboard """

from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data" / "output"
PARQUET_PATH = DATA_DIR / "sales_enriched.parquet"
YEARLY_CSV = DATA_DIR / "yearly.csv"
SEGMENT_CSV = DATA_DIR / "segment_yearly.csv"
REGION_CSV = DATA_DIR / "regional_revenue.csv"
TOP_PRODUCTS_CSV = DATA_DIR / "top_products.csv"


@st.cache_data(show_spinner=False)
def load_dataset() -> pl.DataFrame:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PARQUET_PATH}. Run `python3 src/etl.py` first."
        )
    return pl.read_parquet(PARQUET_PATH)


@st.cache_data(show_spinner=False)
def load_summary(path: Path) -> pl.DataFrame | None:
    if path.exists():
        return pl.read_csv(path)
    return None


def main() -> None:
    st.set_page_config(page_title="Smart Sales Analyzer", layout="wide")
    st.title("📊 Smart Sales Analyzer Dashboard")
    st.caption("Kaggle dataset augmented with 100000 locally generated synthetic rows.")

    try:
        dataset = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    total_rows = len(dataset)
    total_revenue = dataset["sales"].sum()
    unique_customers = dataset["customer_id"].n_unique()
    unique_products = dataset["product_id"].n_unique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{total_rows:,}")
    col2.metric("Total Revenue", f"${total_revenue:,.0f}")
    col3.metric("Unique Customers", f"{unique_customers:,}")
    col4.metric("Unique Products", f"{unique_products:,}")

    yearly = load_summary(YEARLY_CSV)
    segment = load_summary(SEGMENT_CSV)
    regional = load_summary(REGION_CSV)
    top_products = load_summary(TOP_PRODUCTS_CSV)

    with st.container():
        st.subheader("Revenue by Year")
        if yearly is not None and len(yearly) > 0:
            chart = (
                alt.Chart(yearly.to_pandas())
                .mark_bar()
                .encode(
                    x=alt.X("order_year:O", title="Year"),
                    y=alt.Y("revenue:Q", title="Revenue"),
                    tooltip=["order_year", "rows", "revenue", "unique_customers"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Run the ETL pipeline to populate this chart.")

    left, right = st.columns(2)

    with left:
        st.subheader("Revenue by Segment and Year")
        if segment is not None and len(segment) > 0:
            chart = (
                alt.Chart(segment.to_pandas())
                .mark_circle(size=200)
                .encode(
                    x=alt.X("order_year:O", title="Year"),
                    y=alt.Y("segment:N", title="Segment"),
                    color=alt.Color("revenue:Q", scale=alt.Scale(scheme="blues")),
                    size=alt.Size("revenue:Q", legend=None),
                    tooltip=["order_year", "segment", "revenue"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Segment-level data unavailable.")

    with right:
        st.subheader("Top Regions by Revenue")
        if regional is not None and len(regional) > 0:
            chart = (
                alt.Chart(regional.to_pandas())
                .mark_bar()
                .encode(
                    x=alt.X("revenue:Q", title="Revenue"),
                    y=alt.Y("region:N", sort="-x", title="Region"),
                    tooltip=["region", "revenue"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Regional summary unavailable.")

    st.subheader("Top 15 Products")
    if top_products is not None and len(top_products) > 0:
        st.dataframe(top_products.sort("revenue", descending=True).head(15).to_pandas())
    else:
        st.info("Product-level summary unavailable.")


if __name__ == "__main__":
    main()
