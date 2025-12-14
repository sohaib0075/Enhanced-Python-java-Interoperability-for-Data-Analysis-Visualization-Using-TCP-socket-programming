"""
Real-time Analytics Module
Provides streaming data support, live chart updates, and alert system
"""

import threading
import time
import queue
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from collections import deque

# Global state
streaming_active = False
stream_queue = queue.Queue()
data_buffer = deque(maxlen=1000)  # Keep last 1000 data points
alert_callbacks = []
alert_thresholds = {}

def start_streaming(data_source, interval=1.0):
    """
    Start streaming data from a source
    
    Args:
        data_source: Function that returns new data point
        interval: Time between updates (seconds)
    """
    global streaming_active
    
    def stream_worker():
        global streaming_active
        streaming_active = True
        logging.info(f"Streaming started (interval: {interval}s)")
        
        while streaming_active:
            try:
                new_data = data_source()
                if new_data is not None:
                    timestamp = datetime.now()
                    data_point = {
                        'timestamp': timestamp,
                        'data': new_data
                    }
                    stream_queue.put(data_point)
                    data_buffer.append(data_point)
                    logging.debug(f"Streamed data point: {new_data}")
            except Exception as e:
                logging.error(f"Error in streaming: {e}")
            
            time.sleep(interval)
    
    thread = threading.Thread(target=stream_worker, daemon=True)
    thread.start()
    return thread

def stop_streaming():
    """Stop streaming data"""
    global streaming_active
    streaming_active = False
    logging.info("Streaming stopped")

def get_latest_data(n=10):
    """Get latest n data points from buffer"""
    return list(data_buffer)[-n:]

def get_stream_data():
    """Get next data point from stream queue (non-blocking)"""
    try:
        return stream_queue.get_nowait()
    except queue.Empty:
        return None

def register_alert(column, condition, threshold, callback):
    """
    Register an alert
    
    Args:
        column: Column to monitor
        condition: '>', '<', '>=', '<=', '=='
        threshold: Threshold value
        callback: Function to call when alert triggers
    """
    alert_id = f"{column}_{condition}_{threshold}_{len(alert_thresholds)}"
    alert_thresholds[alert_id] = {
        'column': column,
        'condition': condition,
        'threshold': threshold,
        'callback': callback,
        'triggered': False
    }
    logging.info(f"Alert registered: {alert_id}")
    return alert_id

def check_alerts(data_point):
    """Check if any alerts should trigger"""
    if not isinstance(data_point, dict) or 'data' not in data_point:
        return
    
    data = data_point['data']
    if not isinstance(data, (pd.Series, dict)):
        return
    
    for alert_id, alert in alert_thresholds.items():
        if alert['triggered']:
            continue
        
        column = alert['column']
        condition = alert['condition']
        threshold = alert['threshold']
        
        # Get value
        if isinstance(data, pd.Series):
            value = data.get(column, None)
        else:
            value = data.get(column, None)
        
        if value is None:
            continue
        
        # Check condition
        triggered = False
        if condition == '>' and value > threshold:
            triggered = True
        elif condition == '<' and value < threshold:
            triggered = True
        elif condition == '>=' and value >= threshold:
            triggered = True
        elif condition == '<=' and value <= threshold:
            triggered = True
        elif condition == '==' and value == threshold:
            triggered = True
        
        if triggered:
            alert['triggered'] = True
            alert['callback'](alert_id, column, value, threshold, data_point['timestamp'])
            logging.warning(f"Alert triggered: {alert_id} - {column} {condition} {threshold} (value: {value})")

def clear_alert(alert_id):
    """Clear/reset an alert"""
    if alert_id in alert_thresholds:
        alert_thresholds[alert_id]['triggered'] = False
        logging.info(f"Alert cleared: {alert_id}")

def list_alerts():
    """List all registered alerts"""
    return [
        {
            'id': alert_id,
            'column': alert['column'],
            'condition': alert['condition'],
            'threshold': alert['threshold'],
            'triggered': alert['triggered']
        }
        for alert_id, alert in alert_thresholds.items()
    ]

def clear_all_alerts():
    """Clear all alerts"""
    alert_thresholds.clear()
    logging.info("All alerts cleared")

def create_live_chart_data(data_points, chart_type='line'):
    """
    Create chart data from streaming data points
    
    Args:
        data_points: List of data points
        chart_type: 'line', 'scatter', 'bar'
    
    Returns:
        Dictionary with chart data
    """
    if not data_points:
        return None
    
    timestamps = [dp['timestamp'] for dp in data_points]
    
    # Extract numeric values
    values = []
    for dp in data_points:
        data = dp['data']
        if isinstance(data, (pd.Series, dict)):
            # Get first numeric value
            for val in (data.values() if isinstance(data, dict) else data):
                if isinstance(val, (int, float)):
                    values.append(val)
                    break
        elif isinstance(data, (int, float)):
            values.append(data)
    
    return {
        'timestamps': [ts.isoformat() for ts in timestamps],
        'values': values,
        'chart_type': chart_type,
        'count': len(data_points)
    }

# Alert callback example
def default_alert_callback(alert_id, column, value, threshold, timestamp):
    """Default alert callback - logs the alert"""
    message = f"ALERT: {column} = {value} (threshold: {threshold}) at {timestamp}"
    logging.warning(message)
    print(f"\n🚨 {message}\n")





