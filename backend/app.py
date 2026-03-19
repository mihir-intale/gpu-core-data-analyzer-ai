"""
Flask Backend for Analyzer API
Provides /chat endpoint for natural language data analysis
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
import os
import base64
import tempfile

# Add parent directory to path to import analyzer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import BedrockLLM, DataFrameAnalyzer, OllamaLLM, LMStudioLLM, AnthropicLLM
from blob_convertor import file_to_blob

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def denormalize_data(data, jobs):
    """
    Denormalize the nested json data structure into a flat DataFrame.
    
    Structure:
    - Each element in data array represents one row
    - data.values[0] -> values_{job[0]}
    - data.values[1] -> values_{job[1]}
    - data.values[2] -> values_{job[2]}
    
    - data.deltas[0] -> deltas_{job[1]} (offset by 1, no delta for job[0])
    - data.deltas[1] -> deltas_{job[2]}
    
    - data.ratios[0] -> ratios_{job[1]} (offset by 1, no ratio for job[0])
    - data.ratios[1] -> ratios_{job[2]}
    
    Example with 3 jobs:
    Columns: name, taskId, values_{job[0]}, values_{job[1]}, values_{job[2]}, 
             deltas_{job[1]}, deltas_{job[2]}, ratios_{job[1]}, ratios_{job[2]}
    """
    if not data or not jobs:
        return pd.DataFrame()
    
    rows = []
    
    for item in data:
        row = {
            'name': item.get('name', ''),
            'taskId': item.get('taskId', '')
        }
        
        # Process values - each values[i] corresponds to job[i]
        values = item.get('values', [])
        for i, job in enumerate(jobs):
            job_id = job.get('id', i)
            job_name = job.get('name', i).replace(",", "_")
            column_name = f'values_{i} ({job_id}: {job_name})'
            
            if i < len(values):
                # print(column_name, values[i][-1])
                row[column_name] = values[i][-1] # Last iteration
            else:
                row[column_name] = None
        
        # Process deltas - offset by 1 (no delta for job[0])
        deltas = item.get('deltas', [])
        for i, delta in enumerate(deltas):
            # deltas[0] -> job[1], deltas[1] -> job[2], etc.
            job_index = i + 1
            if job_index < len(jobs):
                job_name = jobs[job_index].get('name', f'job_{jobs[job_index].get("id", job_index)}')
                column_name = f'deltas_{job_name}'
                row[column_name] = delta
        
        # Process ratios - offset by 1 (no ratio for job[0])
        ratios = item.get('ratios', [])
        for i, ratio in enumerate(ratios):
            # ratios[0] -> job[1], ratios[1] -> job[2], etc.
            job_index = i + 1
            if job_index < len(jobs):
                job_name = jobs[job_index].get('name', f'job_{jobs[job_index].get("id", job_index)}')
                column_name = f'ratios_{job_name}'
                row[column_name] = ratio
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def process_job_details(data, jobs):
    """
    Process job details data to create scalar and vector DataFrames.
    
    Args:
        data: List of job details arrays, where each array contains job results
              data[0] corresponds to jobs[0], data[1] to jobs[1], etc.
              
              Structure can be:
              1. data[job][iteration][{name, scalars, vectors}]  (original format)
              2. data[job][workload][iteration][{name, scalars, vectors}]  (nested format)
              
              Each job details dictionary contains:
              - name: workload name
              - scalars: list of scalar metrics
              - vectors: list of vector metrics
        jobs: List of job metadata [{"id": job_id, "name": job_name}, ...]
    
    Returns:
        tuple: (scalar_df, vector_df)
            - scalar_df: DataFrame with workload name, job_id, and all scalar metrics as columns
            - vector_df: DataFrame with workload name, timestamp, job_id, and vector metrics as columns
    """
    if not data or not jobs:
        return pd.DataFrame(), pd.DataFrame()
    
    # Process scalars
    scalar_rows = []
    
    for job_idx, job_details_array in enumerate(data):
        if not job_details_array or not isinstance(job_details_array, list):
            continue
        
        # Get job info
        job_info = jobs[job_idx] if job_idx < len(jobs) else {'id': job_idx, 'name': f'job_{job_idx}'}
        job_id = job_info.get('id', job_idx)
        job_name = job_info.get('name', f'job_{job_idx}')
        
        # Handle nested structure: data[job][workload][iteration]
        # Check if first element is a list (nested structure)
        if len(job_details_array) > 0 and isinstance(job_details_array[0], list):
            # Nested format: iterate through workloads
            for workload_array in job_details_array:
                if not isinstance(workload_array, list) or len(workload_array) == 0:
                    continue
                
                # Get the last iteration for this workload
                job_detail = workload_array[-1]
                
                if not isinstance(job_detail, dict):
                    continue
                
                # Process this workload's scalars
                scalar_row = process_scalar_row(job_detail, job_id, job_name)
                if scalar_row:
                    scalar_rows.append(scalar_row)
        else:
            # Original format: data[job][iteration]
            # Get the last element from the array
            job_detail = job_details_array[-1]
            
            if not isinstance(job_detail, dict):
                continue
            
            # Process scalars
            scalar_row = process_scalar_row(job_detail, job_id, job_name)
            if scalar_row:
                scalar_rows.append(scalar_row)
    
    scalar_df = pd.DataFrame(scalar_rows)
    
    # Process vectors
    vector_rows = []
    
    for job_idx, job_details_array in enumerate(data):
        if not job_details_array or not isinstance(job_details_array, list):
            continue
        
        # Get job info
        job_info = jobs[job_idx] if job_idx < len(jobs) else {'id': job_idx, 'name': f'job_{job_idx}'}
        job_id = job_info.get('id', job_idx)
        job_name = job_info.get('name', f'job_{job_idx}')
        
        # Handle nested structure: data[job][workload][iteration]
        if len(job_details_array) > 0 and isinstance(job_details_array[0], list):
            # Nested format: iterate through workloads
            for workload_array in job_details_array:
                if not isinstance(workload_array, list) or len(workload_array) == 0:
                    continue
                
                # Get the last iteration for this workload
                job_detail = workload_array[-1]
                
                if not isinstance(job_detail, dict):
                    continue
                
                # Process this workload's vectors
                workload_vector_rows = process_vector_rows(job_detail, job_id, job_name)
                vector_rows.extend(workload_vector_rows)
        else:
            # Original format: data[job][iteration]
            job_detail = job_details_array[-1]
            
            if not isinstance(job_detail, dict):
                continue
            
            # Process vectors
            workload_vector_rows = process_vector_rows(job_detail, job_id, job_name)
            vector_rows.extend(workload_vector_rows)
    
    vector_df = pd.DataFrame(vector_rows)
    
    return scalar_df, vector_df


def process_scalar_row(job_detail, job_id, job_name):
    """
    Process a single job detail dictionary to extract scalar metrics.
    
    Args:
        job_detail: Dictionary with {name, scalars, vectors, ...}
        job_id: Job ID
        job_name: Job name
        
    Returns:
        Dictionary with scalar metrics or None if no scalars
    """
    workload_name = job_detail.get('name', '')
    scalars = job_detail.get('scalars', [])
    
    if not scalars:
        return None
    
    # Create a row for this job's scalars
    scalar_row = {
        'workload_name': workload_name,
        'job_id': f"{job_id}: {job_name}"
    }
    
    # Add all scalar metrics as columns
    for scalar in scalars:
        if not isinstance(scalar, dict):
            continue
        
        metric_name = scalar.get('metricName', 'unknown')
        tool_name = scalar.get('toolName', '')
        value = scalar.get('value')
        
        # Create column name with tool and metric
        col_name = f"{tool_name}_{metric_name}" if tool_name else metric_name
        scalar_row[col_name] = value
    
    return scalar_row


def process_vector_rows(job_detail, job_id, job_name):
    """
    Process a single job detail dictionary to extract vector metrics.
    
    Args:
        job_detail: Dictionary with {name, scalars, vectors, ...}
        job_id: Job ID
        job_name: Job name
        
    Returns:
        List of dictionaries with vector data rows
    """
    workload_name = job_detail.get('name', '')
    vectors = job_detail.get('vectors', [])
    
    if not vectors:
        return []
    
    # Find the Frame times vector to use as the base timestamp
    frame_times_vector = None
    for vector in vectors:
        if vector.get('metricName') == 'Frame times':
            frame_times_vector = vector
            break
    
    if not frame_times_vector:
        return []
    
    # Get frame time values (timestamps)
    frame_values = frame_times_vector.get('values', [])
    
    vector_rows = []
    
    # Create rows for each timestamp
    for idx, frame_val in enumerate(frame_values):
        if not isinstance(frame_val, dict):
            continue
        
        # OX_value is the timestamp (in ms), OY_value is the frame time
        timestamp = frame_val.get('OX_value', idx)
        
        vector_row = {
            'workload_name': workload_name,
            'timestamp_ms': timestamp,
            'job_id': f"{job_id}: {job_name}"
        }
        
        # Add all vector values at this index
        for vector in vectors:
            if not isinstance(vector, dict):
                continue
            
            metric_name = vector.get('metricName', 'unknown')
            tool_name = vector.get('toolName', '')
            values_list = vector.get('values', [])
            
            # Get value at current index
            if idx < len(values_list):
                val_obj = values_list[idx]
                if isinstance(val_obj, dict):
                    # Use OY_value as the actual metric value
                    value = val_obj.get('OY_value')
                else:
                    value = val_obj
            else:
                value = None
            
            # Create column name
            col_name = f"{tool_name}_{metric_name}" if tool_name else metric_name
            vector_row[col_name] = value
        
        vector_rows.append(vector_row)
    
    return vector_rows


def convert_result_to_blob(result):
    """
    Convert DataFrame or plot result to blob format for JSON serialization
    
    Args:
        result: Dictionary with 'type' and 'value' keys
        
    Returns:
        Dictionary with blob data if applicable
    """
    if not isinstance(result, dict) or 'type' not in result or 'value' not in result:
        return result
    
    result_type = result['type']
    value = result['value']
    
    if result_type == 'dataframe':
        # Convert DataFrame to CSV blob
        if isinstance(value, pd.DataFrame):
            # Save DataFrame to temporary CSV file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
            temp_path = temp_file.name
            temp_file.close()
            
            try:
                value.to_csv(temp_path, index=False)
                blob = file_to_blob(temp_path)
                
                return {
                    'type': 'dataframe',
                    'value': blob['data'],
                    'metadata': blob['metadata'],
                    'blob_type': 'csv'
                }
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            return result
            
    elif result_type == 'plot':
        # Convert plot image to blob
        if isinstance(value, str) and os.path.exists(value):
            blob = file_to_blob(value)
            
            return {
                'type': 'plot',
                'value': blob['data'],
                'metadata': blob['metadata'],
                'blob_type': 'image'
            }
        else:
            return result
    elif result_type == 'html':
        # Plotly HTML — pass through as-is (already a string)
        return {
            'type': 'html',
            'value': value,
            'blob_type': 'html'
        }
    else:
        # For string, number, error types, return as-is
        return result


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint for data analysis
    
    Expected JSON format:
    {
        "data": [...],      # List of data records
        "jobs": [...],      # Additional job data (optional)
        "message": "..."    # Natural language query
    }
    
    Returns:
    {
        "success": true/false,
        "response": "...",   # Analysis result
        "error": "..."       # Error message (if any)
    }
    """
    try:
        # Get request data
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Extract fields
        data = request_data.get('data', [])
        jobs = request_data.get('jobs', [])
        message = request_data.get('message', '')
        
        # Validate inputs
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        if not data and not jobs:
            return jsonify({
                'success': False,
                'error': 'Either data or jobs must be provided'
            }), 400
        # Denormalize the data structure
        df = denormalize_data(data, jobs)
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'DataFrame is empty'
            }), 400
    
        # Save for debugging
        df.to_csv('temp_data.csv', index=False)
        
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"💬 Message: {message}")
        
        # Create Ollama LLM instance
        # llm = OllamaLLM(
        #     model="gpt-oss:20b",
        #     base_url="http://localhost:11434"
        # )
        llm = LMStudioLLM(
            model="openai/gpt-oss-20b",
            base_url="http://localhost:1234"
        )
        
        # Prepare dataframes list with descriptions
        dataframes = [
            {
                "dataframe": df,
                "dataframe_description": "GPU performance metrics across different driver versions and games. Contains values, deltas (performance differences), and ratios (performance ratios) for each metric."
            }
        ]
        
        # Create analyzer with the dataframes list and LLM
        analyzer = DataFrameAnalyzer(dataframes, llm=llm, use_sandbox=True)
            
        # Get response from analyzer
        print("🤖 Processing query...")
        response = analyzer.chat(message)
        analyzer.clear_history()

        # Convert result to blob if it's a DataFrame or plot
        result = response.get('result', {})
        result_blob = convert_result_to_blob(result)
        
        # Response now contains: {"code": "...", "result": {...}, "reply": "..."}
        return jsonify({
            'code': response.get('code', ''),
            'result': result_blob,
            'reply': response.get('reply', '')
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/chat_v2', methods=['POST'])
def chat_v2():
    """
    Chat endpoint for data analysis with job details format (scalars + vectors)
    
    Expected JSON format:
    {
        "data": [          # List of job details arrays
            [{...}, {...}],  # Job 1 results (array of results, last one is used)
            [{...}, {...}]   # Job 2 results (array of results, last one is used)
        ],
        "jobs": [          # Job metadata
            {"id": 123, "name": "baseline"},
            {"id": 456, "name": "optimized"}
        ],
        "message": "..."   # Natural language query
    }
    
    Each job detail object contains:
    {
        "name": "workload_name",
        "scalars": [{metricName, toolName, value, ...}, ...],
        "vectors": [{metricName, toolName, values: [{OX_value, OY_value}, ...], ...}, ...]
    }
    
    Returns:
    {
        "code": "...",     # Generated Python code
        "result": {...},   # Analysis result (blob format)
        "reply": "..."     # LLM-generated reply
    }
    """
    DEBUG=True
    try:
        # Get request data
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Extract fields
        data = request_data.get('data', [])
        jobs = request_data.get('jobs', [])
        message = request_data.get('message', '')
        
        # Validate inputs
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Data is required'
            }), 400
        
        # Process job details to create scalar and vector DataFrames
        scalar_df, vector_df = process_job_details(data, jobs)
        if DEBUG:
            scalar_df.to_csv('debug_scalar_df.csv', index=False)
            vector_df.to_csv('debug_vector_df.csv', index=False)
        
        
        if scalar_df.empty and vector_df.empty:
            return jsonify({
                'success': False,
                'error': 'Both scalar and vector DataFrames are empty'
            }), 400
        
        # Save for debugging
        if not scalar_df.empty:
            scalar_df.to_csv('temp_scalar_data.csv', index=False)
            print(f"📊 Scalar DataFrame shape: {scalar_df.shape}")
            print(f"📋 Scalar Columns: {list(scalar_df.columns)[:10]}...")  # Show first 10 columns
        
        if not vector_df.empty:
            vector_df.to_csv('temp_vector_data.csv', index=False)
            print(f"📊 Vector DataFrame shape: {vector_df.shape}")
            print(f"📋 Vector Columns: {list(vector_df.columns)[:10]}...")  # Show first 10 columns
        
        print(f"💬 Message: {message}")
        
        # Create Ollama LLM instance
        # llm = OllamaLLM(
        #     model="gpt-oss:20b",
        #     base_url="http://localhost:11434"
        # )
        # llm = LMStudioLLM(
        #     model="openai/gpt-oss-20b",
        #     base_url="http://localhost:1234"
        # )
        # llm = BedrockLLM(
        #     model="claude-4-sonnet", 
        #     base_url="https://gnai.intel.com/api/providers/aws/bedrock",
        #     gnai_token="ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SnpkV0lpT2lKdGRHRnNaU0lzSW1GMVpDSTZXeUlpWFN3aVpYaHdJam94TnpjeU5UY3dOelk1TENKelkyOXdaWE1pT2xzaVlYQndPbUZ5ZEdsbVlXTjBiM0o1SWl3aVlYQndPbWQwWVNJc0ltRndjRHBuZEdGNElpd2lZWEJ3T21kdVlXa2lYWDAuZTlvZWF1ZUhjWk8xRzBrS0pYbDdRX1RBS0VDa3VCYjd6R3d3ZzhvZy1Lbw=="
        # )
        llm = AnthropicLLM(
            model="claude-4-5-sonnet",
            base_url="https://gnai.intel.com/api/providers/anthropic"
        )
        
        # Prepare dataframes list with descriptions
        dataframes = []
        
        if not scalar_df.empty:
            dataframes.append({
                "dataframe": scalar_df,
                "dataframe_description": "Scalar performance metrics for each workload and job. Contains aggregated metrics like average FPS, percentile values, max/min values, power consumption, frequency data, etc. Each row represents a workload for a specific job."
            })
        
        if not vector_df.empty:
            dataframes.append({
                "dataframe": vector_df,
                "dataframe_description": "Time-series vector data for each workload and job. Contains frame-by-frame measurements including frame times, latency, frequency changes over time. Each row represents a single timestamp/frame with all measured metrics at that point."
            })
        
        # Create analyzer with the dataframes list and LLM
        analyzer = DataFrameAnalyzer(dataframes, llm=llm, use_sandbox=True)
            
        # Get response from analyzer
        print("🤖 Processing query...")
        response = analyzer.chat(message)
        analyzer.clear_history()

        # Convert result to blob if it's a DataFrame or plot
        result = response.get('result', {})
        result_blob = convert_result_to_blob(result)
        
        # Response now contains: {"code": "...", "result": {...}, "reply": "..."}
        return jsonify({
            'code': response.get('code', ''),
            'result': result_blob,
            'reply': response.get('reply', '')
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Check if Ollama is accessible
        import requests
        
        # Create session with no proxy for localhost
        session = requests.Session()
        session.trust_env = False
        session.proxies = {'http': None, 'https': None}
        
        response = session.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = "running" if response.status_code == 200 else "unreachable"
    except:
        ollama_status = "unreachable"
    
    return jsonify({
        'status': 'healthy',
        'ollama': ollama_status,
        'model': 'gpt-oss:20b'
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        'name': 'Analyzer API',
        'version': '2.0.0',
        'endpoints': {
            '/chat': 'POST - Submit data and message for analysis (legacy format)',
            '/chat_v2': 'POST - Submit job details with scalars and vectors for analysis',
            '/health': 'GET - Check API health status',
            '/': 'GET - This information page'
        },
        'chat_example': {
            'url': '/chat',
            'method': 'POST',
            'body': {
                'data': [
                    {'name': 'Alice', 'age': 25, 'salary': 50000},
                    {'name': 'Bob', 'age': 30, 'salary': 60000}
                ],
                'jobs': [],
                'message': 'What is the average salary?'
            }
        },
        'chat_v2_example': {
            'url': '/chat_v2',
            'method': 'POST',
            'description': 'Processes job details with scalar and vector metrics',
            'body': {
                'data': [
                    [{'name': 'workload1', 'scalars': [...], 'vectors': [...]}],
                    [{'name': 'workload1', 'scalars': [...], 'vectors': [...]}]
                ],
                'jobs': [
                    {'id': 123, 'name': 'baseline'},
                    {'id': 456, 'name': 'optimized'}
                ],
                'message': 'Compare FPS between baseline and optimized'
            }
        }
    }), 200


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Analyzer Flask Backend")
    print("=" * 60)
    print("📡 API Endpoints:")
    print("   • POST /chat    - Submit analysis queries")
    print("   • GET  /health  - Health check")
    print("   • GET  /        - API information")
    print("=" * 60)
    print("🤖 Using Ollama model: gpt-oss:20b")
    print("🔗 Ollama URL: http://localhost:11434")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
