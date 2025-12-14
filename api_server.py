"""
REST API Wrapper for Python Data Visualization Server
Provides HTTP endpoints for web-based access
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os
import sys
import base64
import pandas as pd
from pathlib import Path

# Make sure local folder is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# IMPORTANT: import the server module (single shared state)
import server

app = Flask(__name__)
CORS(app)  # Enable CORS for web clients


# ---------------- Advanced Command Processor ----------------

def process_advanced_command(cmd: str) -> str:
    """Process advanced commands (cleaning, transformations, etc.)"""
    if server.df is None or not isinstance(server.df, pd.DataFrame) or server.df.empty:
        return "Error: Dataset not loaded. Please load a dataset first."
    
    cmd = cmd.strip().lower()
    
    # Data cleaning commands
    if cmd in {"remove_duplicates", "drop_duplicates"}:
        original_len = len(server.df)
        server.df = server.df.drop_duplicates()
        removed = original_len - len(server.df)
        return f"Removed {removed} duplicate rows. {len(server.df)} rows remaining."
    
    elif cmd in {"dropna", "remove_missing"}:
        original_len = len(server.df)
        server.df = server.df.dropna()
        removed = original_len - len(server.df)
        return f"Removed {removed} rows with missing values. {len(server.df)} rows remaining."
    
    elif cmd.startswith("fillna "):
        try:
            parts = cmd.split()
            if len(parts) < 2:
                return "Error: Please specify fill value. Usage: fillna <value> or fillna mean/median/mode"
            
            fill_value = parts[1].lower()
            if fill_value == "mean":
                server.df = server.df.fillna(server.df.select_dtypes(include="number").mean())
                return "Missing values filled with mean"
            elif fill_value == "median":
                server.df = server.df.fillna(server.df.select_dtypes(include="number").median())
                return "Missing values filled with median"
            elif fill_value == "mode":
                server.df = server.df.fillna(server.df.mode().iloc[0])
                return "Missing values filled with mode"
            else:
                try:
                    value = float(fill_value) if "." in fill_value else int(fill_value)
                    server.df = server.df.fillna(value)
                    return f"Missing values filled with {value}"
                except ValueError:
                    return f"Error: Invalid fill value '{fill_value}'. Use: mean, median, mode, or a number"
        except Exception as e:
            return f"Error: {e}"
    
    elif cmd in {"clean_data", "clean"}:
        original_len = len(server.df)
        server.df = server.df.drop_duplicates()
        server.df = server.df.dropna(how="all")
        numeric_cols = server.df.select_dtypes(include="number").columns
        server.df[numeric_cols] = server.df[numeric_cols].fillna(server.df[numeric_cols].mean())
        server.df = server.df.ffill().bfill()
        removed = original_len - len(server.df)
        return f"Data cleaned: {removed} rows removed, {len(server.df)} rows remaining"
    
    # Data transformation commands
    elif cmd == "normalize" or cmd.startswith("normalize "):
        numeric_cols = server.df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return "Error: No numeric columns to normalize"
        
        for col in numeric_cols:
            col_min = server.df[col].min()
            col_max = server.df[col].max()
            if col_max != col_min:
                server.df[col] = (server.df[col] - col_min) / (col_max - col_min)
            else:
                server.df[col] = 0
        
        return f"Normalized {len(numeric_cols)} numeric columns (Min-Max scaling)"
    
    elif cmd in {"standardize", "zscore"}:
        numeric_cols = server.df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return "Error: No numeric columns to standardize"
        
        for col in numeric_cols:
            col_mean = server.df[col].mean()
            col_std = server.df[col].std()
            if col_std != 0:
                server.df[col] = (server.df[col] - col_mean) / col_std
            else:
                server.df[col] = 0
        
        return f"Standardized {len(numeric_cols)} numeric columns (Z-score)"
    
    elif cmd.startswith("log_transform "):
        try:
            parts = cmd.split()
            col_idx = int(parts[1])
            
            if col_idx >= len(server.df.columns):
                return f"Error: Column index {col_idx} out of range"
            
            col = server.df.columns[col_idx]
            if server.df[col].dtype not in ["int64", "float64"]:
                return f"Error: Column {col_idx} is not numeric"
            
            import numpy as np
            server.df[col] = server.df[col].apply(lambda x: np.log(x + 1) if x >= 0 else np.nan)
            return f"Applied log transformation to column {col_idx}"
        except (ValueError, IndexError):
            return "Error: Invalid command. Use 'log_transform <column_index>'"
    
    # ML Commands
    elif cmd.startswith(("ml_train_regression ", "train_regression ")):
        try:
            from ml_operations import train_regression_model
            parts = cmd.split()
            if len(parts) < 2:
                return "Error: Usage: ml_train_regression <target_col> [model_type] [feature_cols]"
            target_col = int(parts[1])
            model_type = parts[2] if len(parts) > 2 else "linear"
            feature_cols = [int(x) for x in parts[3:]] if len(parts) > 3 else None
            result = train_regression_model(server.df, target_col, feature_cols, model_type)
            return f"Model trained: {result['model_name']}\nTest R²: {result['test_r2']:.4f}\nTest MSE: {result['test_mse']:.4f}"
        except Exception as e:
            return f"Error training model: {e}"
    
    elif cmd.startswith(("ml_train_classification ", "train_classification ")):
        try:
            from ml_operations import train_classification_model
            parts = cmd.split()
            if len(parts) < 2:
                return "Error: Usage: ml_train_classification <target_col> [model_type]"
            target_col = int(parts[1])
            model_type = parts[2] if len(parts) > 2 else "logistic"
            result = train_classification_model(server.df, target_col, model_type)
            return f"Model trained: {result['model_name']}\nTest Accuracy: {result['test_accuracy']:.4f}"
        except Exception as e:
            return f"Error training model: {e}"
    
    elif cmd.startswith("ml_predict "):
        try:
            from ml_operations import predict
            parts = cmd.split()
            if len(parts) < 3:
                return "Error: Usage: ml_predict <model_name> <feature1> <feature2> ..."
            model_name = parts[1]
            features = [float(x) for x in parts[2:]]
            prediction = predict(model_name, features)
            return f"Prediction: {prediction}"
        except Exception as e:
            return f"Error making prediction: {e}"
    
    elif cmd.startswith("ml_evaluate "):
        try:
            from ml_operations import evaluate_model
            parts = cmd.split()
            model_name = parts[1] if len(parts) > 1 else None
            if not model_name:
                return "Error: Usage: ml_evaluate <model_name>"
            result = evaluate_model(model_name, server.df)
            return result
        except Exception as e:
            return f"Error evaluating model: {e}"
    
    elif cmd.startswith("ml_importance "):
        try:
            from ml_operations import feature_importance
            parts = cmd.split()
            model_name = parts[1] if len(parts) > 1 else None
            if not model_name:
                return "Error: Usage: ml_importance <model_name>"
            importances = feature_importance(model_name)
            return importances
        except Exception as e:
            return f"Error getting feature importance: {e}"
    
    elif cmd == "ml_list":
        try:
            from ml_operations import list_models
            models = list_models()
            return models if models else "No models trained yet"
        except Exception as e:
            return f"Error listing models: {e}"
    
    # Real-time Analytics Commands
    elif cmd.startswith("stream_start "):
        try:
            from realtime_analytics import start_streaming
            parts = cmd.split()
            interval = float(parts[1]) if len(parts) > 1 else 1.0
            def data_source():
                if isinstance(server.df, pd.DataFrame) and not server.df.empty:
                    return server.df.iloc[-1].to_dict()
                return {}
            start_streaming(data_source, interval)
            return f"Streaming started with interval {interval} seconds"
        except Exception as e:
            return f"Error starting stream: {e}"
    
    elif cmd == "stream_stop":
        try:
            from realtime_analytics import stop_streaming
            stop_streaming()
            return "Streaming stopped"
        except Exception as e:
            return f"Error stopping stream: {e}"
    
    elif cmd.startswith("stream_get "):
        try:
            from realtime_analytics import get_latest_data
            parts = cmd.split()
            n = int(parts[1]) if len(parts) > 1 else 10
            data_points = get_latest_data(n)
            result = f"Latest {len(data_points)} data points:\n"
            for dp in data_points:
                result += f"  {dp['timestamp']}: {dp['data']}\n"
            return result
        except Exception as e:
            return f"Error getting data: {e}"
    
    elif cmd.startswith("alert_set "):
        try:
            from realtime_analytics import register_alert, default_alert_callback
            parts = cmd.split()
            if len(parts) < 4:
                return "Error: Usage: alert_set <column> <condition> <threshold>"
            column = int(parts[1])
            condition = parts[2]
            threshold = float(parts[3])
            alert_id = register_alert(column, condition, threshold, default_alert_callback)
            return f"Alert registered: {alert_id}"
        except Exception as e:
            return f"Error setting alert: {e}"
    
    elif cmd == "alert_list":
        try:
            from realtime_analytics import list_alerts
            alerts = list_alerts()
            return alerts if alerts else "No alerts set"
        except Exception as e:
            return f"Error listing alerts: {e}"
    
    elif cmd.startswith("alert_clear "):
        try:
            from realtime_analytics import clear_alert
            parts = cmd.split()
            alert_id = parts[1] if len(parts) > 1 else None
            if alert_id:
                clear_alert(alert_id)
                return f"Alert cleared: {alert_id}"
            return "Error: Please provide alert ID"
        except Exception as e:
            return f"Error clearing alert: {e}"
    
    # Database Commands
    elif cmd.startswith("db_save "):
        try:
            from database import save_dataset_to_db
            dataset_name = cmd[8:].strip()
            if not dataset_name:
                return "Error: Please provide a dataset name. Usage: db_save <name>"
            dataset_id = save_dataset_to_db(dataset_name, server.df, "Saved from API")
            return f"Dataset saved to database as '{dataset_name}' (ID: {dataset_id})"
        except Exception as e:
            return f"Error saving dataset: {e}"
    
    elif cmd.startswith("db_load "):
        try:
            from database import load_dataset_from_db
            dataset_name = cmd[8:].strip()
            if not dataset_name:
                return "Error: Please provide a dataset name. Usage: db_load <name>"
            server.df = load_dataset_from_db(dataset_name)
            return f"Dataset '{dataset_name}' loaded from database with shape {server.df.shape}"
        except Exception as e:
            return f"Error loading dataset: {e}"
    
    elif cmd == "db_list":
        try:
            from database import list_datasets
            datasets = list_datasets()
            return datasets if datasets else "No datasets in database"
        except Exception as e:
            return f"Error listing datasets: {e}"
    
    elif cmd.startswith("db_delete "):
        try:
            from database import delete_dataset
            dataset_name = cmd[10:].strip()
            if not dataset_name:
                return "Error: Please provide a dataset name. Usage: db_delete <name>"
            if delete_dataset(dataset_name):
                return f"Dataset '{dataset_name}' deleted from database"
            return f"Error: Dataset '{dataset_name}' not found"
        except Exception as e:
            return f"Error deleting dataset: {e}"
    
    elif cmd.startswith("sql "):
        try:
            from database import execute_sql_query
            query = cmd[4:].strip()
            if not query:
                return "Error: Please provide a SQL query. Usage: sql SELECT ..."
            result_df = execute_sql_query(query)
            return result_df.to_string()
        except Exception as e:
            return f"Error executing SQL query: {e}"
    
    # Chart customization commands
    elif cmd.startswith("chart_title "):
        title = cmd[11:].strip()
        server.set_chart_config(title=title)
        return f"Chart title set to: {title}"
    
    elif cmd.startswith("chart_xlabel "):
        xlabel = cmd[13:].strip()
        server.set_chart_config(xlabel=xlabel)
        return f"X-axis label set to: {xlabel}"
    
    elif cmd.startswith("chart_ylabel "):
        ylabel = cmd[13:].strip()
        server.set_chart_config(ylabel=ylabel)
        return f"Y-axis label set to: {ylabel}"
    
    elif cmd.startswith("chart_color "):
        color = cmd[12:].strip()
        server.set_chart_config(color=color)
        return f"Chart color set to: {color}"
    
    elif cmd.startswith("chart_size "):
        try:
            size_str = cmd[11:].strip()
            width, height = map(float, size_str.split(","))
            server.set_chart_config(figsize=(width, height))
            return f"Chart size set to: {width}x{height}"
        except (ValueError, IndexError):
            return "Error: Invalid size format. Use: chart_size <width>,<height>"
    
    elif cmd.startswith("chart_dpi "):
        try:
            dpi = int(cmd[10:].strip())
            server.set_chart_config(dpi=dpi)
            return f"Chart DPI set to: {dpi}"
        except ValueError:
            return "Error: Invalid DPI value. Use: chart_dpi <number>"
    
    elif cmd in {"chart_reset", "reset_chart"}:
        server.reset_chart_config()
        return "Chart configuration reset to defaults"
    
    # Fallback to safe dataframe commands
    return server.safe_execute_dataframe_command(cmd)


# ---------------- Health ----------------

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Server is running"})


# ---------------- Load dataset ----------------

@app.route("/api/load", methods=["POST"])
def load_data():
    """Load dataset from file"""
    try:
        data = request.get_json(silent=True) or {}
        file_path = data.get("file_path")

        if not file_path:
            return jsonify({"error": "file_path is required"}), 400

        if not server.validate_file_path(file_path):
            return jsonify({"error": "Invalid file path. Access denied for security reasons."}), 403

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return jsonify({"error": f"File '{file_path}' not found"}), 404

        file_size = file_path_obj.stat().st_size
        if file_size > server.MAX_FILE_SIZE:
            return jsonify({
                "error": f"File too large (max {server.MAX_FILE_SIZE / 1024 / 1024}MB)"
            }), 400

        # Load into shared server.df
        server.df = pd.read_csv(file_path, sep=None, engine="python", header=None)

        return jsonify({
            "success": True,
            "message": "Dataset loaded successfully",
            "shape": list(server.df.shape),
            "columns": server.df.columns.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Get data ----------------

@app.route("/api/data", methods=["GET"])
def get_data():
    """Get dataset as JSON"""
    if server.df is None or not isinstance(server.df, pd.DataFrame) or server.df.empty:
        return jsonify({"error": "Dataset not loaded"}), 400

    try:
        limit = request.args.get("limit", type=int)
        offset = request.args.get("offset", 0, type=int)

        data_df = server.df
        if limit:
            data_df = server.df.iloc[offset: offset + limit]

        return jsonify({
            "columns": server.df.columns.tolist(),
            "data": data_df.values.tolist(),
            "shape": list(server.df.shape),
            "returned_rows": len(data_df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Summary ----------------

@app.route("/api/summary", methods=["GET"])
def get_summary():
    """Get statistical summary"""
    if server.df is None or not isinstance(server.df, pd.DataFrame) or server.df.empty:
        return jsonify({"error": "Dataset not loaded"}), 400

    try:
        summary = server.df.describe()
        return jsonify({
            "summary": summary.to_dict(),
            "shape": list(server.df.shape)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Command executor ----------------

@app.route("/api/command", methods=["POST"])
def execute_command():
    """Execute a command (text or visualization)"""
    try:
        data = request.get_json(silent=True) or {}
        cmd = (data.get("command") or "").strip().lower()

        if not cmd:
            return jsonify({"error": "command is required"}), 400

        # If user sends a filename, load it
        if cmd.endswith(".txt") or cmd.endswith(".csv"):
            if not server.validate_file_path(cmd):
                return jsonify({"error": "Invalid file path"}), 403

            file_path = Path(cmd)
            if not file_path.exists():
                return jsonify({"error": "File not found"}), 404

            server.df = pd.read_csv(cmd, sep=None, engine="python", header=None)
            return jsonify({
                "success": True,
                "message": "Dataset loaded",
                "shape": list(server.df.shape)
            })

        # Visualization commands
        viz_commands = {
            "chart": server.make_regression_chart,
            "regression": server.make_regression_chart,
            "violin": server.make_violin_chart,
            "pair": server.make_pairplot,
            "pairplot": server.make_pairplot,
            "histogram": server.make_histogram,
            "boxplot": server.make_boxplot,
            "heatmap": server.make_heatmap,
            "barchart": server.make_barchart,
            "bar": server.make_barchart,
            "scattermatrix": server.make_scatter_matrix,
            "scatter_matrix": server.make_scatter_matrix,
        }

        if cmd in viz_commands:
            img_bytes = viz_commands[cmd]()  # reads server.df
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            return jsonify({
                "success": True,
                "type": "image",
                "data": img_base64,
                "format": "jpg"
            })

        # Text summary quick command
        if cmd == "summary":
            if server.df is not None and not server.df.empty:
                return jsonify({
                    "success": True,
                    "type": "text",
                    "data": server.df.describe().to_string()
                })
            return jsonify({"error": "Dataset not loaded"}), 400

        # Process advanced commands (data cleaning, transformations, etc.)
        result = process_advanced_command(cmd)
        if result.startswith("Error:"):
            return jsonify({"error": result}), 400
        
        return jsonify({
            "success": True,
            "type": "text",
            "data": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Visualization endpoint ----------------

@app.route("/api/visualization/<chart_type>", methods=["GET"])
def get_visualization(chart_type):
    """Get visualization as raw JPG image"""
    if server.df is None or not isinstance(server.df, pd.DataFrame) or server.df.empty:
        return jsonify({"error": "Dataset not loaded"}), 400

    chart_type = chart_type.lower()

    viz_functions = {
        "regression": server.make_regression_chart,
        "chart": server.make_regression_chart,
        "violin": server.make_violin_chart,
        "pair": server.make_pairplot,
        "pairplot": server.make_pairplot,
        "histogram": server.make_histogram,
        "boxplot": server.make_boxplot,
        "heatmap": server.make_heatmap,
        "barchart": server.make_barchart,
        "bar": server.make_barchart,
        "scattermatrix": server.make_scatter_matrix,
        "scatter_matrix": server.make_scatter_matrix,
    }

    if chart_type not in viz_functions:
        return jsonify({"error": f"Unknown chart type: {chart_type}"}), 400

    try:
        img_bytes = viz_functions[chart_type]()
        return send_file(
            io.BytesIO(img_bytes),
            mimetype="image/jpeg",
            as_attachment=False
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Info ----------------

@app.route("/api/info", methods=["GET"])
def get_info():
    """Get server and dataset information"""
    dataset_loaded = (
        server.df is not None and
        isinstance(server.df, pd.DataFrame) and
        not server.df.empty
    )

    info = {
        "server": "Python Data Visualization API",
        "version": "1.0",
        "dataset_loaded": dataset_loaded
    }

    if dataset_loaded:
        info["dataset_shape"] = list(server.df.shape)
        info["dataset_columns"] = server.df.columns.tolist()
        info["dataset_dtypes"] = {str(k): str(v) for k, v in server.df.dtypes.items()}

    return jsonify(info)


# ---------------- Run ----------------

if __name__ == "__main__":
    print("Starting REST API server on http://127.0.0.1:5000")
    print("API Documentation:")
    print("  GET  /api/health - Health check")
    print("  POST /api/load - Load dataset")
    print("  GET  /api/data - Get dataset")
    print("  GET  /api/summary - Get statistical summary")
    print("  POST /api/command - Execute command")
    print("  GET  /api/visualization/<type> - Get visualization")
    print("  GET  /api/info - Get server info")

    app.run(host="127.0.0.1", port=5000, debug=True)



