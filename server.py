import socket
import sys
import os
import io
import logging
import re
import threading
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------------- Configuration ----------------

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "server": {
        "host": "127.0.0.1",
        "port": 1234,
        "max_connections": 5,
        "timeout_seconds": 30
    },
    "security": {
        "allowed_base_dir": ".",
        "max_file_size_mb": 50,
        "max_command_length": 1000
    },
    "logging": {
        "level": "INFO",
        "log_file": "server.log"
    },
    "data": {
        "default_dataset": None
    }
}


def load_config():
    """Load configuration from file or use defaults."""
    config_path = Path(CONFIG_FILE)
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_from_file = json.load(f)

            merged = DEFAULT_CONFIG.copy()
            for key, value in config_from_file.items():
                if isinstance(value, dict) and key in merged:
                    merged[key].update(value)
                else:
                    merged[key] = value
            return merged

        except Exception as e:
            print(f"Warning: Could not load config file: {e}. Using defaults.")

    return DEFAULT_CONFIG


config = load_config()

HOST = config["server"]["host"]
PORT = config["server"]["port"]
MAX_CONNECTIONS = config["server"]["max_connections"]
TIMEOUT_SECONDS = config["server"]["timeout_seconds"]

ALLOWED_BASE_DIR = Path(config["security"]["allowed_base_dir"]).resolve()
if not ALLOWED_BASE_DIR.exists():
    ALLOWED_BASE_DIR = Path.cwd()

MAX_FILE_SIZE = config["security"]["max_file_size_mb"] * 1024 * 1024
MAX_COMMAND_LENGTH = config["security"]["max_command_length"]

LOG_LEVEL = getattr(logging, config["logging"]["level"], logging.INFO)
LOG_FILE = config["logging"]["log_file"]

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

df = None  # loaded dataset


# ---------------- Helpers ----------------

def send_text(sock, text: str):
    """Safely send text to client."""
    try:
        sock.sendall(text.encode("utf-8"))
    except Exception as e:
        logging.error(f"Error sending text: {e}")


def _get_dataframe_info(dframe: pd.DataFrame) -> str:
    """Capture dataframe info as string."""
    buf = io.StringIO()
    dframe.info(buf=buf)
    return buf.getvalue()


def validate_file_path(file_path: str) -> bool:
    """Validate file path to prevent directory traversal attacks."""
    try:
        abs_path = Path(file_path).resolve()
        # Python 3.9+: is_relative_to
        return abs_path.is_relative_to(ALLOWED_BASE_DIR.resolve())
    except Exception:
        return False


def safe_execute_dataframe_command(cmd: str) -> str:
    """
    Safely execute dataframe operations without using eval().
    Supports a whitelist of safe operations.
    """
    global df

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "Error: Dataset not loaded. Please load a dataset first."

    cmd = cmd.strip().lower()

    safe_commands = {
        "shape": lambda: str(df.shape),
        "columns": lambda: str(list(df.columns)),
        "dtypes": lambda: df.dtypes.to_string(),
        "head": lambda: df.head(10).to_string(),
        "tail": lambda: df.tail(10).to_string(),
        "info": lambda: _get_dataframe_info(df),
        "isnull": lambda: df.isnull().sum().to_string(),
        "nunique": lambda: df.nunique().to_string(),
        "mean": lambda: (
            df.select_dtypes(include="number").mean().to_string()
            if not df.select_dtypes(include="number").empty else "No numeric columns"
        ),
        "median": lambda: (
            df.select_dtypes(include="number").median().to_string()
            if not df.select_dtypes(include="number").empty else "No numeric columns"
        ),
        "std": lambda: (
            df.select_dtypes(include="number").std().to_string()
            if not df.select_dtypes(include="number").empty else "No numeric columns"
        ),
        "var": lambda: (
            df.select_dtypes(include="number").var().to_string()
            if not df.select_dtypes(include="number").empty else "No numeric columns"
        ),
        "variance": lambda: (
            df.select_dtypes(include="number").var().to_string()
            if not df.select_dtypes(include="number").empty else "No numeric columns"
        ),
        "min": lambda: (
            df.select_dtypes(include="number").min().to_string()
            if not df.select_dtypes(include="number").empty else "No numeric columns"
        ),
        "max": lambda: (
            df.select_dtypes(include="number").max().to_string()
            if not df.select_dtypes(include="number").empty else "No numeric columns"
        ),
        "corr": lambda: (
            df.select_dtypes(include="number").corr().to_string()
            if len(df.select_dtypes(include="number").columns) >= 2
            else "Need at least 2 numeric columns for correlation"
        ),
        "correlation": lambda: (
            df.select_dtypes(include="number").corr().to_string()
            if len(df.select_dtypes(include="number").columns) >= 2
            else "Need at least 2 numeric columns for correlation"
        ),
        "skew": lambda: (
            df.select_dtypes(include="number").skew().to_string()
            if not df.select_dtypes(include="number").empty else "No numeric columns"
        ),
        "kurtosis": lambda: (
            df.select_dtypes(include="number").kurtosis().to_string()
            if not df.select_dtypes(include="number").empty else "No numeric columns"
        ),
    }

    # head n
    if cmd.startswith("head "):
        try:
            n = int(cmd.split()[1])
            return df.head(min(n, 100)).to_string()
        except (ValueError, IndexError):
            return "Error: Invalid head command. Use 'head <number>'"

    # tail n
    if cmd.startswith("tail "):
        try:
            n = int(cmd.split()[1])
            return df.tail(min(n, 100)).to_string()
        except (ValueError, IndexError):
            return "Error: Invalid tail command. Use 'tail <number>'"

    # filter
    if cmd.startswith("filter "):
        try:
            parts = cmd.split()
            if len(parts) < 4:
                return (
                    "Error: Invalid filter command. Use "
                    "'filter <column> <operator> <value>'\n"
                    "Operators: >, <, >=, <=, ==, !="
                )

            col_idx = int(parts[1])
            operator = parts[2]
            value = float(parts[3]) if "." in parts[3] else int(parts[3])

            if col_idx >= len(df.columns):
                return (
                    f"Error: Column index {col_idx} out of range. "
                    f"Dataset has {len(df.columns)} columns."
                )

            col = df.columns[col_idx]
            if operator == ">":
                filtered = df[df[col] > value]
            elif operator == "<":
                filtered = df[df[col] < value]
            elif operator == ">=":
                filtered = df[df[col] >= value]
            elif operator == "<=":
                filtered = df[df[col] <= value]
            elif operator == "==":
                filtered = df[df[col] == value]
            elif operator == "!=":
                filtered = df[df[col] != value]
            else:
                return (
                    f"Error: Invalid operator '{operator}'. "
                    "Use: >, <, >=, <=, ==, !="
                )

            original_len = len(df)
            df = filtered.copy()
            return (
                f"Filtered dataset: {len(filtered)} rows remaining "
                f"(from {original_len} rows)"
            )

        except (ValueError, IndexError) as e:
            return f"Error: Invalid filter command. {e}"

    # sort
    if cmd.startswith("sort "):
        try:
            parts = cmd.split()
            if len(parts) < 2:
                return "Error: Invalid sort command. Use 'sort <column> [asc/desc]'"

            col_idx = int(parts[1])
            ascending = True
            if len(parts) > 2:
                ascending = parts[2].lower() != "desc"

            if col_idx >= len(df.columns):
                return (
                    f"Error: Column index {col_idx} out of range. "
                    f"Dataset has {len(df.columns)} columns."
                )

            col = df.columns[col_idx]
            df = df.sort_values(by=col, ascending=ascending)
            return (
                f"Dataset sorted by column {col_idx} "
                f"({'ascending' if ascending else 'descending'})"
            )
        except (ValueError, IndexError) as e:
            return f"Error: Invalid sort command. {e}"

    # group
    if cmd.startswith("group "):
        try:
            parts = cmd.split()
            if len(parts) < 2:
                return "Error: Invalid group command. Use 'group <column> [mean/sum/count]'"

            col_idx = int(parts[1])
            operation = parts[2].lower() if len(parts) > 2 else "mean"

            if col_idx >= len(df.columns):
                return (
                    f"Error: Column index {col_idx} out of range. "
                    f"Dataset has {len(df.columns)} columns."
                )

            group_col = df.columns[col_idx]
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols) == 0:
                return "Error: No numeric columns to aggregate"

            if operation == "mean":
                grouped = df.groupby(group_col)[num_cols].mean()
            elif operation == "sum":
                grouped = df.groupby(group_col)[num_cols].sum()
            elif operation == "count":
                grouped = df.groupby(group_col).size()
                return f"Grouped by {group_col} (count):\n{grouped.to_string()}"
            else:
                return f"Error: Invalid operation '{operation}'. Use: mean, sum, count"

            return f"Grouped by {group_col} ({operation}):\n{grouped.to_string()}"

        except (ValueError, IndexError) as e:
            return f"Error: Invalid group command. {e}"

    # execute safe command
    if cmd in safe_commands:
        try:
            result = safe_commands[cmd]()
            return str(result) if result is not None else "Command executed successfully"
        except Exception as e:
            return f"Error executing command: {e}"

    available = ", ".join(sorted(safe_commands.keys()))
    return (
        f"Error: Unknown command '{cmd}'. "
        f"Available commands: {available}, head <n>, tail <n>"
    )


# ---------------- Chart customization ----------------

chart_config = {
    "title": None,
    "xlabel": None,
    "ylabel": None,
    "figsize": None,
    "dpi": 120,
    "color": None,
    "style": "default"
}


def set_chart_config(**kwargs):
    """Set chart customization options."""
    global chart_config
    chart_config.update({k: v for k, v in kwargs.items() if v is not None})


def reset_chart_config():
    """Reset chart configuration to defaults."""
    global chart_config
    chart_config = {
        "title": None,
        "xlabel": None,
        "ylabel": None,
        "figsize": None,
        "dpi": 120,
        "color": None,
        "style": "default"
    }


# ---------- Figure: Matplotlib regression (price vs size) ----------

def make_regression_chart() -> bytes:
    global df
    figsize = chart_config.get("figsize") or (8, 6)
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(df, pd.DataFrame) and not df.empty:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) >= 3:
            X = df[[num_cols[0]]].values  # size
            y = df[num_cols[2]].values    # price

            color = chart_config.get("color") or "blue"
            ax.scatter(X, y, label="Data", color=color, alpha=0.6)

            model = LinearRegression().fit(X, y)
            xs = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
            ax.plot(xs, model.predict(xs), linewidth=2, label="Regression line", color="red")

            title = chart_config.get("title") or "Linear Regression: Price vs Size"
            ax.set_title(title)

            xlabel = chart_config.get("xlabel") or f"{num_cols[0]} (sq.ft.)"
            ylabel = chart_config.get("ylabel") or f"{num_cols[2]} ($)"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Insufficient numeric columns", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "Dataset not loaded", ha="center", va="center")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="jpg", dpi=chart_config.get("dpi", 120))
    plt.close(fig)
    return buf.getvalue()


# ---------- Figure: Seaborn violin (rooms vs price) ----------

def make_violin_chart() -> bytes:
    global df
    fig, ax = plt.subplots()

    if isinstance(df, pd.DataFrame) and not df.empty:
        df.columns = ["size", "rooms", "price"]
        sns.violinplot(x="rooms", y="price", data=df, ax=ax, inner="quartile")
        ax.set_title("Violin Plot: Rooms vs Price")
    else:
        ax.text(0.5, 0.5, "Dataset not loaded", ha="center", va="center")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="jpg", dpi=120)
    plt.close(fig)
    return buf.getvalue()


# ---------- Figure: Seaborn pairplot (correlations) ----------

def make_pairplot() -> bytes:
    global df

    if isinstance(df, pd.DataFrame) and not df.empty:
        df.columns = ["size", "rooms", "price"]
        g = sns.pairplot(df)
        buf = io.BytesIO()
        g.savefig(buf, format="jpg", dpi=120)
        plt.close("all")
        return buf.getvalue()

    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "Dataset not loaded", ha="center", va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=120)
    plt.close(fig)
    return buf.getvalue()


# ---------- Figure: Histogram ----------

def make_histogram() -> bytes:
    global df
    fig, ax = plt.subplots()

    if isinstance(df, pd.DataFrame) and not df.empty:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            col = num_cols[0]
            ax.hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
            ax.set_title(f"Histogram: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "Dataset not loaded", ha="center", va="center")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="jpg", dpi=120)
    plt.close(fig)
    return buf.getvalue()


# ---------- Figure: Boxplot ----------

def make_boxplot() -> bytes:
    global df
    fig, ax = plt.subplots()

    if isinstance(df, pd.DataFrame) and not df.empty:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            data_to_plot = [df[col].dropna() for col in num_cols[:5]]
            labels = [str(col) for col in num_cols[:5]]
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")

            ax.set_title("Boxplot: Distribution of Numeric Columns")
            ax.set_ylabel("Values")
            ax.set_xlabel("Columns")
            ax.grid(True, alpha=0.3, axis="y")
            plt.xticks(rotation=45, ha="right")
        else:
            ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "Dataset not loaded", ha="center", va="center")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="jpg", dpi=120)
    plt.close(fig)
    return buf.getvalue()


# ---------- Figure: Correlation Heatmap ----------

def make_heatmap() -> bytes:
    global df
    fig, ax = plt.subplots(figsize=(10, 8))

    if isinstance(df, pd.DataFrame) and not df.empty:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr()
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=1,
                ax=ax,
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title("Correlation Heatmap")
        else:
            ax.text(0.5, 0.5, "Need at least 2 numeric columns for correlation",
                    ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "Dataset not loaded", ha="center", va="center")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="jpg", dpi=120)
    plt.close(fig)
    return buf.getvalue()


# ---------- Figure: Bar Chart ----------

def make_barchart() -> bytes:
    global df
    fig, ax = plt.subplots()

    if isinstance(df, pd.DataFrame) and not df.empty:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            col = num_cols[0]
            value_counts = df[col].value_counts().head(20)

            ax.bar(range(len(value_counts)), value_counts.values,
                   color="steelblue", edgecolor="black")
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels([str(x) for x in value_counts.index],
                               rotation=45, ha="right")

            ax.set_title(f"Bar Chart: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "Dataset not loaded", ha="center", va="center")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="jpg", dpi=120)
    plt.close(fig)
    return buf.getvalue()


# ---------- Figure: Scatter Matrix ----------

def make_scatter_matrix() -> bytes:
    global df

    if not (isinstance(df, pd.DataFrame) and not df.empty):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Dataset not loaded", ha="center", va="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=120)
        plt.close(fig)
        return buf.getvalue()

    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need at least 2 numeric columns for scatter matrix",
                ha="center", va="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=120)
        plt.close(fig)
        return buf.getvalue()

    cols_to_plot = num_cols[:5].tolist()
    if len(cols_to_plot) < 2:
        cols_to_plot = num_cols.tolist()[:2]

    n = len(cols_to_plot)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))

    for i, col1 in enumerate(cols_to_plot):
        for j, col2 in enumerate(cols_to_plot):
            ax = axes[i][j]

            if i == j:
                ax.hist(df[col1].dropna(), bins=20, edgecolor="black", alpha=0.7)
            else:
                ax.scatter(df[col2], df[col1], alpha=0.5, s=20)

            if i == n - 1:
                ax.set_xlabel(col2)
            if j == 0:
                ax.set_ylabel(col1)

    plt.suptitle("Scatter Matrix", y=0.995, fontsize=14)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="jpg", dpi=120)
    plt.close(fig)
    return buf.getvalue()


# ---------------- Client handler ----------------

def handle_client(clt, adr):
    """Handle a single client connection."""
    global df

    try:
        clt.settimeout(TIMEOUT_SECONDS)

        data = clt.recv(24576)
        if not data:
            logging.warning(f"No data received from {adr}")
            return

        try:
            cmd = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            send_text(clt, "Error: Invalid encoding. Please send UTF-8 text.")
            logging.error(f"Invalid encoding from {adr}")
            return

        if len(cmd) > MAX_COMMAND_LENGTH:
            send_text(clt, f"Error: Command too long (max {MAX_COMMAND_LENGTH} characters)")
            logging.warning(f"Command too long from {adr}")
            return

        logging.info(f"Command from {adr}: {cmd}")

        # exit
        if cmd in {"exit()", "quit()"}:
            send_text(clt, "Server shutting down.")
            logging.info("Server shutdown requested")
            sys.exit(0)

        # file loading
        elif cmd.endswith(".txt") or cmd.endswith(".csv"):
            if not validate_file_path(cmd):
                send_text(clt, "Error: Invalid file path. Access denied for security reasons.")
                logging.warning(f"Invalid file path attempt from {adr}: {cmd}")
                return

            file_path = Path(cmd)
            if not file_path.exists():
                send_text(clt, f"Error: File '{cmd}' not found")
                logging.warning(f"File not found: {cmd}")
                return

            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                send_text(clt, f"Error: File too large (max {MAX_FILE_SIZE / 1024 / 1024}MB)")
                logging.warning(f"File too large: {cmd} ({file_size} bytes)")
                return

            try:
                df = pd.read_csv(cmd, sep=None, engine="python", header=None)
                send_text(clt, f"DataFrame 'df' loaded from '{cmd}' with shape {df.shape}")
                logging.info(f"Dataset loaded: {cmd}, shape: {df.shape}")
            except pd.errors.EmptyDataError:
                send_text(clt, f"Error: File '{cmd}' is empty")
                logging.error(f"Empty file: {cmd}")
            except Exception as e:
                send_text(clt, f"Error loading file: {e}")
                logging.error(f"Error loading file {cmd}: {e}", exc_info=True)

        # visualization commands
        elif cmd == "chart":
            try:
                clt.sendall(make_regression_chart())
                logging.info(f"Regression chart sent to {adr}")
            except Exception as e:
                send_text(clt, f"Error generating chart: {e}")
                logging.error("Error generating chart", exc_info=True)

        elif cmd == "violin":
            try:
                clt.sendall(make_violin_chart())
                logging.info(f"Violin chart sent to {adr}")
            except Exception as e:
                send_text(clt, f"Error generating violin plot: {e}")
                logging.error("Error generating violin plot", exc_info=True)

        elif cmd == "pair":
            try:
                clt.sendall(make_pairplot())
                logging.info(f"Pairplot sent to {adr}")
            except Exception as e:
                send_text(clt, f"Error generating pairplot: {e}")
                logging.error("Error generating pairplot", exc_info=True)

        elif cmd == "histogram":
            try:
                clt.sendall(make_histogram())
                logging.info(f"Histogram sent to {adr}")
            except Exception as e:
                send_text(clt, f"Error generating histogram: {e}")
                logging.error("Error generating histogram", exc_info=True)

        elif cmd == "boxplot":
            try:
                clt.sendall(make_boxplot())
                logging.info(f"Boxplot sent to {adr}")
            except Exception as e:
                send_text(clt, f"Error generating boxplot: {e}")
                logging.error("Error generating boxplot", exc_info=True)

        elif cmd == "heatmap":
            try:
                clt.sendall(make_heatmap())
                logging.info(f"Heatmap sent to {adr}")
            except Exception as e:
                send_text(clt, f"Error generating heatmap: {e}")
                logging.error("Error generating heatmap", exc_info=True)

        elif cmd in {"barchart", "bar"}:
            try:
                clt.sendall(make_barchart())
                logging.info(f"Bar chart sent to {adr}")
            except Exception as e:
                send_text(clt, f"Error generating bar chart: {e}")
                logging.error("Error generating bar chart", exc_info=True)

        elif cmd in {"scattermatrix", "scatter_matrix"}:
            try:
                clt.sendall(make_scatter_matrix())
                logging.info(f"Scatter matrix sent to {adr}")
            except Exception as e:
                send_text(clt, f"Error generating scatter matrix: {e}")
                logging.error("Error generating scatter matrix", exc_info=True)

        # summary
        elif cmd == "summary":
            if isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    summary = df.describe().to_string()
                    send_text(clt, summary)
                    logging.info(f"Summary sent to {adr}")
                except Exception as e:
                    send_text(clt, f"Error generating summary: {e}")
                    logging.error("Error generating summary", exc_info=True)
            else:
                send_text(clt, "Error: Dataset not loaded. Please load a dataset first.")

        # JSON data for UI/table
        elif cmd in {"getdata", "get_data"}:
            if isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    data_dict = {
                        "columns": df.columns.tolist(),
                        "data": df.values.tolist(),
                        "shape": list(df.shape)
                    }
                    send_text(clt, json.dumps(data_dict))
                    logging.info(f"Data sent to {adr} (shape: {df.shape})")
                except Exception as e:
                    send_text(clt, f"Error retrieving data: {e}")
                    logging.error("Error retrieving data", exc_info=True)
            else:
                send_text(clt, "Error: Dataset not loaded. Please load a dataset first.")

        # database operations
        elif cmd.startswith("db_save "):
            try:
                from database import save_dataset_to_db
                dataset_name = cmd[8:].strip()
                if not dataset_name:
                    send_text(clt, "Error: Please provide a dataset name. Usage: db_save <name>")
                else:
                    dataset_id = save_dataset_to_db(dataset_name, df, "Saved from server")
                    send_text(clt, f"Dataset saved to database as '{dataset_name}' (ID: {dataset_id})")
                    logging.info(f"Dataset saved to database: {dataset_name}")
            except Exception as e:
                send_text(clt, f"Error saving to database: {e}")
                logging.error("Error saving to database", exc_info=True)

        elif cmd.startswith("db_load "):
            try:
                from database import load_dataset_from_db
                dataset_name = cmd[8:].strip()
                if not dataset_name:
                    send_text(clt, "Error: Please provide a dataset name. Usage: db_load <name>")
                else:
                    df = load_dataset_from_db(dataset_name)
                    send_text(clt, f"Dataset '{dataset_name}' loaded from database with shape {df.shape}")
                    logging.info(f"Dataset loaded from database: {dataset_name}")
            except Exception as e:
                send_text(clt, f"Error loading from database: {e}")
                logging.error("Error loading from database", exc_info=True)

        elif cmd == "db_list":
            try:
                from database import list_datasets
                datasets = list_datasets()
                if datasets:
                    result = "Datasets in database:\n"
                    for ds in datasets:
                        result += (
                            f"  - {ds['name']}: {ds['rows']} rows, "
                            f"{ds['columns']} columns ({ds['created_at']})\n"
                        )
                    send_text(clt, result)
                else:
                    send_text(clt, "No datasets found in database.")
            except Exception as e:
                send_text(clt, f"Error listing datasets: {e}")
                logging.error("Error listing datasets", exc_info=True)

        elif cmd.startswith("db_delete "):
            try:
                from database import delete_dataset
                dataset_name = cmd[10:].strip()
                if not dataset_name:
                    send_text(clt, "Error: Please provide a dataset name. Usage: db_delete <name>")
                else:
                    if delete_dataset(dataset_name):
                        send_text(clt, f"Dataset '{dataset_name}' deleted from database")
                    else:
                        send_text(clt, f"Error: Dataset '{dataset_name}' not found")
            except Exception as e:
                send_text(clt, f"Error deleting dataset: {e}")
                logging.error("Error deleting dataset", exc_info=True)

        elif cmd.startswith("sql "):
            try:
                from database import execute_sql_query
                query = cmd[4:].strip()
                if not query:
                    send_text(clt, "Error: Please provide a SQL query. Usage: sql SELECT ...")
                else:
                    result_df = execute_sql_query(query)
                    send_text(clt, result_df.to_string())
                    logging.info(f"SQL query executed: {len(result_df)} rows")
            except Exception as e:
                send_text(clt, f"Error executing SQL query: {e}")
                logging.error("Error executing SQL query", exc_info=True)

        # chart customization
        elif cmd.startswith("chart_title "):
            title = cmd[11:].strip()
            set_chart_config(title=title)
            send_text(clt, f"Chart title set to: {title}")

        elif cmd.startswith("chart_xlabel "):
            xlabel = cmd[13:].strip()
            set_chart_config(xlabel=xlabel)
            send_text(clt, f"X-axis label set to: {xlabel}")

        elif cmd.startswith("chart_ylabel "):
            ylabel = cmd[13:].strip()
            set_chart_config(ylabel=ylabel)
            send_text(clt, f"Y-axis label set to: {ylabel}")

        elif cmd.startswith("chart_color "):
            color = cmd[12:].strip()
            set_chart_config(color=color)
            send_text(clt, f"Chart color set to: {color}")

        elif cmd.startswith("chart_size "):
            try:
                size_str = cmd[11:].strip()
                width, height = map(float, size_str.split(","))
                set_chart_config(figsize=(width, height))
                send_text(clt, f"Chart size set to: {width}x{height}")
            except (ValueError, IndexError):
                send_text(clt, "Error: Invalid size format. Use: chart_size <width>,<height>")

        elif cmd.startswith("chart_dpi "):
            try:
                dpi = int(cmd[10:].strip())
                set_chart_config(dpi=dpi)
                send_text(clt, f"Chart DPI set to: {dpi}")
            except ValueError:
                send_text(clt, "Error: Invalid DPI value. Use: chart_dpi <number>")

        elif cmd in {"chart_reset", "reset_chart"}:
            reset_chart_config()
            send_text(clt, "Chart configuration reset to defaults")

        # ML commands
        elif cmd.startswith(("ml_train_regression ", "train_regression ")):
            try:
                from ml_operations import train_regression_model
                parts = cmd.split()

                if len(parts) < 2:
                    send_text(
                        clt,
                        "Error: Usage: ml_train_regression <target_col> [model_type] [feature_cols]\n"
                        "Example: ml_train_regression 2 linear"
                    )
                    return

                target_col = int(parts[1])
                model_type = parts[2] if len(parts) > 2 else "linear"
                feature_cols = [int(x) for x in parts[3:]] if len(parts) > 3 else None

                result = train_regression_model(df, target_col, feature_cols, model_type)
                send_text(
                    clt,
                    f"Model trained: {result['model_name']}\n"
                    f"Test R²: {result['test_r2']:.4f}\n"
                    f"Test MSE: {result['test_mse']:.4f}\n"
                    f"Train samples: {result['train_samples']}, "
                    f"Test samples: {result['test_samples']}"
                )
                logging.info(f"ML regression model trained: {result['model_name']}")
            except Exception as e:
                send_text(clt, f"Error training model: {e}")
                logging.error("Error training regression model", exc_info=True)

        elif cmd.startswith(("ml_train_classification ", "train_classification ")):
            try:
                from ml_operations import train_classification_model
                parts = cmd.split()

                if len(parts) < 2:
                    send_text(
                        clt,
                        "Error: Usage: ml_train_classification <target_col> [model_type]\n"
                        "Example: ml_train_classification 1 logistic"
                    )
                    return

                target_col = int(parts[1])
                model_type = parts[2] if len(parts) > 2 else "logistic"
                feature_cols = [int(x) for x in parts[3:]] if len(parts) > 3 else None

                result = train_classification_model(df, target_col, feature_cols, model_type)
                send_text(
                    clt,
                    f"Model trained: {result['model_name']}\n"
                    f"Test Accuracy: {result['test_accuracy']:.4f}\n"
                    f"Classes: {result['classes']}\n"
                    f"Train samples: {result['train_samples']}, "
                    f"Test samples: {result['test_samples']}"
                )
                logging.info(f"ML classification model trained: {result['model_name']}")
            except Exception as e:
                send_text(clt, f"Error training model: {e}")
                logging.error("Error training classification model", exc_info=True)

        elif cmd.startswith("ml_predict "):
            try:
                from ml_operations import predict
                parts = cmd.split()

                if len(parts) < 3:
                    send_text(clt, "Error: Usage: ml_predict <model_name> <feature1> <feature2> ...")
                    return

                model_name = parts[1]
                features = [float(x) for x in parts[2:]]
                prediction = predict(model_name, features)
                send_text(clt, f"Prediction: {prediction:.4f}")
            except Exception as e:
                send_text(clt, f"Error making prediction: {e}")
                logging.error("Error making prediction", exc_info=True)

        elif cmd.startswith("ml_evaluate "):
            try:
                from ml_operations import evaluate_model
                parts = cmd.split()
                model_name = parts[1] if len(parts) > 1 else None

                if not model_name:
                    send_text(clt, "Error: Usage: ml_evaluate <model_name>")
                    return

                result = evaluate_model(model_name, df)
                if "accuracy" in result:
                    send_text(clt,
                              f"Model Evaluation:\nAccuracy: {result['accuracy']:.4f}\n"
                              f"Samples: {result['samples']}")
                else:
                    send_text(clt,
                              f"Model Evaluation:\nR²: {result['r2']:.4f}\n"
                              f"RMSE: {result['rmse']:.4f}\n"
                              f"Samples: {result['samples']}")
            except Exception as e:
                send_text(clt, f"Error evaluating model: {e}")
                logging.error("Error evaluating model", exc_info=True)

        elif cmd.startswith("ml_importance "):
            try:
                from ml_operations import feature_importance
                parts = cmd.split()
                model_name = parts[1] if len(parts) > 1 else None

                if not model_name:
                    send_text(clt, "Error: Usage: ml_importance <model_name>")
                    return

                importances = feature_importance(model_name)
                result = "Feature Importances:\n"
                for feature, importance in sorted(importances.items(),
                                                  key=lambda x: x[1],
                                                  reverse=True):
                    result += f"  {feature}: {importance:.4f}\n"
                send_text(clt, result)
            except Exception as e:
                send_text(clt, f"Error getting feature importance: {e}")
                logging.error("Error getting feature importance", exc_info=True)

        elif cmd in {"ml_list", "list_models"}:
            try:
                from ml_operations import list_models
                models = list_models()
                if models:
                    result = "Trained Models:\n"
                    for model in models:
                        result += (
                            f"  - {model['name']}: {model['type']} "
                            f"({model['model_type']}), target col {model['target_col']}\n"
                        )
                    send_text(clt, result)
                else:
                    send_text(clt, "No trained models found.")
            except Exception as e:
                send_text(clt, f"Error listing models: {e}")
                logging.error("Error listing models", exc_info=True)

        # streaming / alerts
        elif cmd.startswith("stream_start "):
            try:
                from realtime_analytics import start_streaming

                parts = cmd.split()
                interval = float(parts[1]) if len(parts) > 1 else 1.0

                def data_source():
                    global df
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        return df.iloc[-1].to_dict()
                    return None

                start_streaming(data_source, interval)
                send_text(clt, f"Streaming started with {interval}s interval")
                logging.info(f"Streaming started: {interval}s interval")
            except Exception as e:
                send_text(clt, f"Error starting stream: {e}")
                logging.error("Error starting stream", exc_info=True)

        elif cmd == "stream_stop":
            try:
                from realtime_analytics import stop_streaming
                stop_streaming()
                send_text(clt, "Streaming stopped")
            except Exception as e:
                send_text(clt, f"Error stopping stream: {e}")

        elif cmd.startswith("stream_get "):
            try:
                from realtime_analytics import get_latest_data
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 10

                data_points = get_latest_data(n)
                result = f"Latest {len(data_points)} data points:\n"
                for dp in data_points:
                    result += f"  {dp['timestamp']}: {dp['data']}\n"
                send_text(clt, result)
            except Exception as e:
                send_text(clt, f"Error getting stream data: {e}")

        elif cmd.startswith("alert_set "):
            try:
                from realtime_analytics import register_alert, default_alert_callback

                parts = cmd.split()
                if len(parts) < 4:
                    send_text(
                        clt,
                        "Error: Usage: alert_set <column> <condition> <threshold>\n"
                        "Example: alert_set 2 > 500000"
                    )
                    return

                column = int(parts[1])
                condition = parts[2]
                threshold = float(parts[3]) if "." in parts[3] else int(parts[3])

                alert_id = register_alert(column, condition, threshold, default_alert_callback)
                send_text(clt, f"Alert registered: {alert_id}\nColumn {column} {condition} {threshold}")
                logging.info(f"Alert registered: {alert_id}")
            except Exception as e:
                send_text(clt, f"Error setting alert: {e}")
                logging.error("Error setting alert", exc_info=True)

        elif cmd == "alert_list":
            try:
                from realtime_analytics import list_alerts
                alerts = list_alerts()
                if alerts:
                    result = "Registered Alerts:\n"
                    for alert in alerts:
                        status = "🔴 TRIGGERED" if alert["triggered"] else "🟢 Active"
                        result += (
                            f"  {alert['id']}: Column {alert['column']} "
                            f"{alert['condition']} {alert['threshold']} [{status}]\n"
                        )
                    send_text(clt, result)
                else:
                    send_text(clt, "No alerts registered.")
            except Exception as e:
                send_text(clt, f"Error listing alerts: {e}")

        elif cmd.startswith("alert_clear "):
            try:
                from realtime_analytics import clear_alert
                parts = cmd.split()
                alert_id = parts[1] if len(parts) > 1 else None

                if alert_id:
                    clear_alert(alert_id)
                    send_text(clt, f"Alert cleared: {alert_id}")
                else:
                    send_text(clt, "Error: Please provide alert ID")
            except Exception as e:
                send_text(clt, f"Error clearing alert: {e}")

        # generate all plots
        elif cmd == "all":
            try:
                plot_data = make_regression_chart()
                violin_data = make_violin_chart()
                pair_data = make_pairplot()
                hist_data = make_histogram()
                box_data = make_boxplot()
                heatmap_data = make_heatmap()
                bar_data = make_barchart()
                scatter_data = make_scatter_matrix()

                with open("plot.jpg", "wb") as f:
                    f.write(plot_data)
                with open("violin.jpg", "wb") as f:
                    f.write(violin_data)
                with open("pair.jpg", "wb") as f:
                    f.write(pair_data)
                with open("histogram.jpg", "wb") as f:
                    f.write(hist_data)
                with open("boxplot.jpg", "wb") as f:
                    f.write(box_data)
                with open("heatmap.jpg", "wb") as f:
                    f.write(heatmap_data)
                with open("barchart.jpg", "wb") as f:
                    f.write(bar_data)
                with open("scattermatrix.jpg", "wb") as f:
                    f.write(scatter_data)

                send_text(
                    clt,
                    "All visualizations generated: plot.jpg, violin.jpg, pair.jpg, "
                    "histogram.jpg, boxplot.jpg, heatmap.jpg, barchart.jpg, scattermatrix.jpg"
                )
                logging.info(f"All visualizations generated for {adr}")

            except Exception as e:
                send_text(clt, f"Error generating visualizations: {e}")
                logging.error("Error generating all visualizations", exc_info=True)

        # data cleaning
        elif cmd in {"remove_duplicates", "drop_duplicates"}:
            if isinstance(df, pd.DataFrame) and not df.empty:
                original_len = len(df)
                df = df.drop_duplicates()
                removed = original_len - len(df)
                send_text(clt, f"Removed {removed} duplicate rows. {len(df)} rows remaining.")
                logging.info(f"Removed {removed} duplicate rows")
            else:
                send_text(clt, "Error: Dataset not loaded")

        elif cmd in {"dropna", "remove_missing"}:
            if isinstance(df, pd.DataFrame) and not df.empty:
                original_len = len(df)
                df = df.dropna()
                removed = original_len - len(df)
                send_text(clt, f"Removed {removed} rows with missing values. {len(df)} rows remaining.")
                logging.info(f"Removed {removed} rows with missing values")
            else:
                send_text(clt, "Error: Dataset not loaded")

        elif cmd.startswith("fillna "):
            try:
                parts = cmd.split()
                if len(parts) < 2:
                    send_text(clt, "Error: Please specify fill value. Usage: fillna <value> or fillna mean/median/mode")
                    return

                fill_value = parts[1].lower()
                if fill_value == "mean":
                    df = df.fillna(df.select_dtypes(include="number").mean())
                    send_text(clt, "Missing values filled with mean")
                elif fill_value == "median":
                    df = df.fillna(df.select_dtypes(include="number").median())
                    send_text(clt, "Missing values filled with median")
                elif fill_value == "mode":
                    df = df.fillna(df.mode().iloc[0])
                    send_text(clt, "Missing values filled with mode")
                else:
                    try:
                        value = float(fill_value) if "." in fill_value else int(fill_value)
                        df = df.fillna(value)
                        send_text(clt, f"Missing values filled with {value}")
                    except ValueError:
                        send_text(clt, f"Error: Invalid fill value '{fill_value}'. Use: mean, median, mode, or a number")

                logging.info(f"Missing values filled using: {fill_value}")
            except Exception as e:
                send_text(clt, f"Error: {e}")
                logging.error("Error filling missing values", exc_info=True)

        elif cmd in {"clean_data", "clean"}:
            if isinstance(df, pd.DataFrame) and not df.empty:
                original_len = len(df)

                df = df.drop_duplicates()
                df = df.dropna(how="all")

                numeric_cols = df.select_dtypes(include="number").columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

                df = df.ffill().bfill()

                removed = original_len - len(df)
                send_text(clt, f"Data cleaned: {removed} rows removed, {len(df)} rows remaining")
                logging.info(f"Data cleaned: {removed} rows removed")
            else:
                send_text(clt, "Error: Dataset not loaded")

        elif cmd == "normalize" or cmd.startswith("normalize "):
            if isinstance(df, pd.DataFrame) and not df.empty:
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) == 0:
                    send_text(clt, "Error: No numeric columns to normalize")
                    return

                for col in numeric_cols:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    if col_max != col_min:
                        df[col] = (df[col] - col_min) / (col_max - col_min)
                    else:
                        df[col] = 0

                send_text(clt, f"Normalized {len(numeric_cols)} numeric columns (Min-Max scaling)")
                logging.info(f"Data normalized: {len(numeric_cols)} columns")
            else:
                send_text(clt, "Error: Dataset not loaded")

        elif cmd in {"standardize", "zscore"}:
            if isinstance(df, pd.DataFrame) and not df.empty:
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) == 0:
                    send_text(clt, "Error: No numeric columns to standardize")
                    return

                for col in numeric_cols:
                    col_mean = df[col].mean()
                    col_std = df[col].std()
                    if col_std != 0:
                        df[col] = (df[col] - col_mean) / col_std
                    else:
                        df[col] = 0

                send_text(clt, f"Standardized {len(numeric_cols)} numeric columns (Z-score)")
                logging.info(f"Data standardized: {len(numeric_cols)} columns")
            else:
                send_text(clt, "Error: Dataset not loaded")

        elif cmd.startswith("log_transform "):
            try:
                parts = cmd.split()
                col_idx = int(parts[1])

                if col_idx >= len(df.columns):
                    send_text(clt, f"Error: Column index {col_idx} out of range")
                    return

                col = df.columns[col_idx]
                if df[col].dtype not in ["int64", "float64"]:
                    send_text(clt, f"Error: Column {col_idx} is not numeric")
                    return

                df[col] = df[col].apply(lambda x: np.log(x + 1) if x >= 0 else np.nan)
                send_text(clt, f"Applied log transformation to column {col_idx}")
                logging.info(f"Log transformation applied to column {col_idx}")

            except (ValueError, IndexError):
                send_text(clt, "Error: Invalid command. Use 'log_transform <column_index>'")

        # fallback safe commands
        else:
            result = safe_execute_dataframe_command(cmd)
            send_text(clt, result)

    except socket.timeout:
        send_text(clt, "Error: Connection timeout")
        logging.warning(f"Connection timeout with {adr}")

    except Exception as e:
        send_text(clt, f"Error: {e}")
        logging.error(f"Error handling client {adr}: {e}", exc_info=True)

    finally:
        try:
            clt.close()
        except Exception:
            pass


# ---------------- Main server loop ----------------

def main():
    """Main server function with threading support for concurrent clients."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(MAX_CONNECTIONS)

            logging.info(f"Server listening on {HOST}:{PORT} ...")
            print(f"Server listening on {HOST}:{PORT} ...")
            print("Server supports concurrent client connections.")

            while True:
                try:
                    clt, adr = s.accept()
                    logging.info(f"Client connected from {adr}")

                    client_thread = threading.Thread(
                        target=handle_client,
                        args=(clt, adr),
                        daemon=True
                    )
                    client_thread.start()
                    logging.debug(f"Started thread for client {adr}")

                except KeyboardInterrupt:
                    raise

                except Exception as e:
                    logging.error(f"Error accepting connection: {e}", exc_info=True)

        except OSError as e:
            logging.error(f"Server error: {e}")
            print(f"Error: Could not start server. {e}")
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
        print("\nServer stopped.")
    except Exception as e:
        logging.critical(f"Fatal error: {e}", exc_info=True)
        print(f"Fatal error: {e}")
        sys.exit(1)
