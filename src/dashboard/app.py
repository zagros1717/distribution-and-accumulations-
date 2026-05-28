from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_ROOT = Path(os.environ.get("DATA_ROOT", "/app/data"))

st.set_page_config(
    page_title="BTC Research Machine",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Small filesystem / parquet helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def _glob_files(path_str: str, pattern: str = "part-*.parquet") -> list[str]:
    path = Path(path_str)
    if not path.exists():
        return []
    return [str(p) for p in sorted(path.glob(pattern))]


@st.cache_data(ttl=10)
def read_parquet_dir(path_str: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    files = _glob_files(path_str)
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    return df


@st.cache_data(ttl=10)
def file_count_and_size(path_str: str) -> tuple[int, int]:
    path = Path(path_str)
    if not path.exists():
        return 0, 0
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.1f} {u}" if u != "B" else f"{int(x)} B"
        x /= 1024
    return f"{x:.1f} TB"


def parse_partition_value(name: str, prefix: str) -> str:
    return name.split("=", 1)[1] if name.startswith(prefix + "=") else name


def available_exchanges(root: Path) -> list[str]:
    norm = root / "normalized"
    raw = root / "raw"
    values: set[str] = set()
    if norm.exists():
        values.update(parse_partition_value(p.name, "exchange") for p in norm.iterdir() if p.is_dir())
    if raw.exists():
        values.update(p.name for p in raw.iterdir() if p.is_dir())
    return sorted(values)


def available_symbols(root: Path, exchange: str) -> list[str]:
    values: set[str] = set()
    norm_ex = root / "normalized" / f"exchange={exchange}"
    raw_ex = root / "raw" / exchange
    if norm_ex.exists():
        values.update(parse_partition_value(p.name, "symbol") for p in norm_ex.iterdir() if p.is_dir())
    if raw_ex.exists():
        values.update(p.name for p in raw_ex.iterdir() if p.is_dir())
    return sorted(values)


def available_dates(root: Path, exchange: str, symbol: str) -> list[str]:
    values: set[str] = set()
    paths = [
        root / "raw" / exchange / symbol,
        root / "normalized" / f"exchange={exchange}" / f"symbol={symbol}",
        root / "snapshots" / f"exchange={exchange}" / f"symbol={symbol}",
        root / "features" / "interval_ms=1000" / f"exchange={exchange}" / f"symbol={symbol}",
    ]
    for base in paths:
        if base.exists():
            values.update(parse_partition_value(p.name, "date") for p in base.iterdir() if p.is_dir() and p.name.startswith("date="))
    return sorted(values)


def date_to_dt(date_iso: str) -> datetime:
    return datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def raw_path(root: Path, exchange: str, symbol: str, date_iso: str) -> Path:
    return root / "raw" / exchange / symbol / f"date={date_iso}"


def normalized_path(root: Path, exchange: str, symbol: str, date_iso: str) -> Path:
    return root / "normalized" / f"exchange={exchange}" / f"symbol={symbol}" / f"date={date_iso}"


def snapshots_path(root: Path, exchange: str, symbol: str, date_iso: str, interval_ms: int) -> Path:
    return root / "snapshots" / f"exchange={exchange}" / f"symbol={symbol}" / f"date={date_iso}" / f"interval_ms={interval_ms}"


def features_path(root: Path, exchange: str, symbol: str, date_iso: str, interval_ms: int) -> Path:
    return root / "features" / f"interval_ms={interval_ms}" / f"exchange={exchange}" / f"symbol={symbol}" / f"date={date_iso}"


def labels_path(root: Path, exchange: str, symbol: str, date_iso: str, interval_ms: int, horizon_s: int) -> Path:
    return root / "labels" / f"interval_ms={interval_ms}" / f"horizon_s={horizon_s}" / f"exchange={exchange}" / f"symbol={symbol}" / f"date={date_iso}"


def latest_ts(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns and not df.empty:
            try:
                ts = pd.to_datetime(df[c], utc=True, errors="coerce").dropna()
                if len(ts):
                    return str(ts.max())
            except Exception:
                pass
    return "—"


def show_metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.title("₿ BTC Research Machine Dashboard")
st.caption("Local dashboard for recorder status, data quality, snapshots, model outputs, and reports. No trading execution.")

with st.sidebar:
    st.header("Settings")
    data_root_str = st.text_input("Data root", value=str(DEFAULT_ROOT))
    root = Path(data_root_str).expanduser().resolve()
    st.caption(f"Resolved: `{root}`")

    exchanges = available_exchanges(root)
    if not exchanges:
        st.warning("No data found yet. Start the recorder first.")
        st.code("docker compose up recorder", language="bash")
        st.stop()

    exchange = st.selectbox("Exchange", exchanges, index=0)
    symbols = available_symbols(root, exchange)
    if not symbols:
        st.warning("No symbols found for this exchange.")
        st.stop()
    symbol = st.selectbox("Symbol", symbols, index=0)

    dates = available_dates(root, exchange, symbol)
    if not dates:
        st.warning("No date partitions found for this market.")
        st.stop()
    date_iso = st.selectbox("Date", dates, index=len(dates) - 1)

    interval_ms = st.selectbox("Interval", [1000, 500, 100], index=0)
    horizon_s = st.selectbox("Horizon", [5, 1, 10, 30], index=0)
    max_rows = st.slider("Max rows to load per chart", 1_000, 100_000, 20_000, step=1_000)

    if st.button("Refresh cache"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

raw_dir = raw_path(root, exchange, symbol, date_iso)
norm_dir = normalized_path(root, exchange, symbol, date_iso)
snap_dir = snapshots_path(root, exchange, symbol, date_iso, interval_ms)
feat_dir = features_path(root, exchange, symbol, date_iso, interval_ms)
lab_dir = labels_path(root, exchange, symbol, date_iso, interval_ms, horizon_s)

raw_df = read_parquet_dir(str(raw_dir), max_rows=max_rows)
norm_df = read_parquet_dir(str(norm_dir), max_rows=max_rows)
snap_df = read_parquet_dir(str(snap_dir), max_rows=max_rows)
feat_df = read_parquet_dir(str(feat_dir), max_rows=max_rows)
lab_df = read_parquet_dir(str(lab_dir), max_rows=max_rows)

all_files, all_size = file_count_and_size(str(root))

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

st.subheader(f"Overview — {exchange}/{symbol} — {date_iso}")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    show_metric_card("Raw rows", f"{len(raw_df):,}", "Rows loaded from the selected raw partition, capped by Max rows.")
with c2:
    show_metric_card("Normalized rows", f"{len(norm_df):,}")
with c3:
    show_metric_card("Snapshot rows", f"{len(snap_df):,}")
with c4:
    show_metric_card("Feature rows", f"{len(feat_df):,}")
with c5:
    show_metric_card("Data size", fmt_bytes(all_size), f"{all_files:,} files under data root")

c1, c2, c3 = st.columns(3)
with c1:
    st.info(f"Latest raw receive time: **{latest_ts(raw_df, ['receive_time'])}**")
with c2:
    st.info(f"Latest event time: **{latest_ts(norm_df, ['event_time', 'receive_time'])}**")
with c3:
    st.info(f"Latest snapshot time: **{latest_ts(snap_df, ['ts'])}**")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_status, tab_book, tab_features, tab_model, tab_reports, tab_files = st.tabs([
    "Status",
    "Order book",
    "Features & labels",
    "Model/backtest",
    "Reports",
    "Files",
])

with tab_status:
    st.markdown("### Recorder / data status")
    st.write("Use this tab to check whether live data is being captured and normalized.")

    if raw_df.empty and norm_df.empty:
        st.warning("No raw or normalized rows loaded for the selected date.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Recent raw frames")
            show_cols = [c for c in ["receive_time", "exchange", "symbol", "channel", "payload"] if c in raw_df.columns]
            st.dataframe(raw_df[show_cols].tail(20), use_container_width=True, height=420)
        with col_b:
            st.markdown("#### Recent normalized events")
            show_cols = [c for c in ["event_index", "event_time", "event_type", "side", "price", "size", "sequence", "order_id"] if c in norm_df.columns]
            st.dataframe(norm_df[show_cols].tail(30), use_container_width=True, height=420)

    st.markdown("#### Event type distribution")
    if "event_type" in norm_df.columns and not norm_df.empty:
        counts = norm_df["event_type"].value_counts().reset_index()
        counts.columns = ["event_type", "count"]
        st.plotly_chart(px.bar(counts, x="event_type", y="count", title="Normalized event types"), use_container_width=True)
    else:
        st.caption("No normalized event_type column found yet.")

    log_path = root / "run.log"
    st.markdown("#### run.log tail")
    if log_path.exists():
        try:
            tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-120:])
            st.code(tail or "run.log is empty", language="text")
        except Exception as exc:
            st.error(f"Could not read run.log: {exc}")
    else:
        st.caption("No run.log found yet.")

with tab_book:
    st.markdown("### Order book snapshots")
    if snap_df.empty:
        st.warning("No snapshots found. Run replay or pipeline first.")
        st.code(
            f"docker compose run --rm btc-research python main.py replay --exchange {exchange} --symbol {symbol} --start {date_iso} --end {date_iso}",
            language="bash",
        )
    else:
        snap_df["ts"] = pd.to_datetime(snap_df["ts"], utc=True, errors="coerce")
        valid_pct = float(snap_df.get("is_valid", pd.Series([True] * len(snap_df))).mean() * 100.0) if len(snap_df) else 0.0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valid snapshots", f"{valid_pct:.2f}%")
        col2.metric("Last mid", f"{snap_df['mid_price'].dropna().iloc[-1]:,.2f}" if "mid_price" in snap_df and snap_df["mid_price"].dropna().size else "—")
        col3.metric("Last spread", f"{snap_df['spread'].dropna().iloc[-1]:.2f}" if "spread" in snap_df and snap_df["spread"].dropna().size else "—")
        col4.metric("Rows loaded", f"{len(snap_df):,}")

        if {"ts", "mid_price"}.issubset(snap_df.columns):
            st.plotly_chart(px.line(snap_df, x="ts", y="mid_price", title="Mid price"), use_container_width=True)
        if {"ts", "spread"}.issubset(snap_df.columns):
            st.plotly_chart(px.line(snap_df, x="ts", y="spread", title="Spread"), use_container_width=True)

        depth_cols = [c for c in ["bid_size_1", "ask_size_1", "bid_size_5", "ask_size_5", "bid_size_10", "ask_size_10"] if c in snap_df.columns]
        if "ts" in snap_df.columns and depth_cols:
            st.plotly_chart(px.line(snap_df, x="ts", y=depth_cols, title="Top-level depth"), use_container_width=True)

        st.markdown("#### Snapshot sample")
        st.dataframe(snap_df.tail(100), use_container_width=True)

with tab_features:
    st.markdown("### Features & labels")
    if feat_df.empty:
        st.warning("No features found. Run features or pipeline first.")
    else:
        feat_df["ts"] = pd.to_datetime(feat_df["ts"], utc=True, errors="coerce")
        st.markdown("#### Feature preview")
        st.dataframe(feat_df.tail(100), use_container_width=True)

        numeric_cols = [c for c in feat_df.columns if pd.api.types.is_numeric_dtype(feat_df[c])]
        default_cols = [c for c in ["mid_price", "spread", "microprice_minus_mid", "order_book_imbalance_1", "trade_imbalance_1s"] if c in numeric_cols]
        selected_cols = st.multiselect("Feature chart columns", numeric_cols, default=default_cols[:4])
        if selected_cols and "ts" in feat_df.columns:
            st.plotly_chart(px.line(feat_df, x="ts", y=selected_cols, title="Selected features"), use_container_width=True)

    if lab_df.empty:
        st.warning("No labels found for this interval/horizon. Run labels or pipeline first.")
    else:
        st.markdown("#### Label distribution")
        if "label" in lab_df.columns:
            counts = lab_df["label"].value_counts().sort_index().reset_index()
            counts.columns = ["label", "count"]
            st.plotly_chart(px.bar(counts, x="label", y="count", title="Label balance (-1/0/+1)"), use_container_width=True)
        st.dataframe(lab_df.tail(100), use_container_width=True)

with tab_model:
    st.markdown("### Model and OOS predictions")
    model_dir = root / "models" / "xgboost" / f"horizon_{horizon_s}s"
    meta_path = model_dir / "metadata.json"
    oos_path = model_dir / "oos_predictions.parquet"

    if not model_dir.exists():
        st.warning("No model directory found yet. Run train or pipeline first.")
        st.code(
            f"docker compose run --rm btc-research python main.py pipeline --exchange {exchange} --symbol {symbol} --start {date_iso} --end {date_iso} --horizon {horizon_s}",
            language="bash",
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model dir exists", "yes")
        c2.metric("Metadata", "yes" if meta_path.exists() else "no")
        c3.metric("OOS predictions", "yes" if oos_path.exists() else "no")

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                st.markdown("#### Training metadata")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Folds", len(meta.get("folds", [])))
                m2.metric("Features", len(meta.get("feature_columns", [])))
                m3.metric("Horizon", f"{meta.get('horizon_s', horizon_s)}s")
                m4.metric("Interval", f"{meta.get('interval_ms', interval_ms)}ms")

                folds = meta.get("folds", [])
                if folds:
                    fold_df = pd.DataFrame(folds)
                    keep = [c for c in ["fold_index", "train_start", "train_end", "val_start", "val_end", "n_train", "n_val", "accuracy", "log_loss", "precision_long", "precision_short", "n_signals_long", "n_signals_short"] if c in fold_df.columns]
                    st.dataframe(fold_df[keep], use_container_width=True)
                    metric_cols = [c for c in ["accuracy", "log_loss", "precision_long", "precision_short"] if c in fold_df.columns]
                    if metric_cols:
                        st.plotly_chart(px.line(fold_df, x="fold_index", y=metric_cols, markers=True, title="Fold metrics"), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not parse metadata.json: {exc}")

        if oos_path.exists():
            try:
                oos_df = pd.read_parquet(oos_path)
                if len(oos_df) > max_rows:
                    oos_df = oos_df.tail(max_rows).reset_index(drop=True)
                oos_df["ts"] = pd.to_datetime(oos_df["ts"], utc=True, errors="coerce")
                st.markdown("#### OOS prediction probabilities")
                prob_cols = [c for c in ["prob_short", "prob_flat", "prob_long"] if c in oos_df.columns]
                if prob_cols:
                    st.plotly_chart(px.line(oos_df, x="ts", y=prob_cols, title="Out-of-sample probabilities"), use_container_width=True)
                st.dataframe(oos_df.tail(100), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read OOS predictions: {exc}")

with tab_reports:
    st.markdown("### Markdown reports")
    reports_dir = root / "reports"
    reports = sorted(reports_dir.glob("*.md"), reverse=True) if reports_dir.exists() else []
    if not reports:
        st.warning("No reports found yet. Run report or pipeline first.")
    else:
        report = st.selectbox("Report", reports, format_func=lambda p: p.name)
        st.download_button("Download report", report.read_bytes(), file_name=report.name)
        st.markdown(report.read_text(errors="ignore"))

with tab_files:
    st.markdown("### Data folders")
    rows = []
    for p in [raw_dir, norm_dir, snap_dir, feat_dir, lab_dir, root / "models", root / "reports"]:
        n, size = file_count_and_size(str(p))
        rows.append({"path": str(p), "files": n, "size": fmt_bytes(size), "exists": p.exists()})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### Helpful commands")
    st.code(
        f"""# Start recorder
        docker compose up -d recorder

        # Follow recorder logs
        docker compose logs -f recorder

        # Run full pipeline for selected date
        docker compose run --rm btc-research python main.py pipeline --exchange {exchange} --symbol {symbol} --start {date_iso} --end {date_iso} --horizon {horizon_s}

        # Stop everything
        docker compose down
        """.replace("        ", ""),
        language="bash",
    )
