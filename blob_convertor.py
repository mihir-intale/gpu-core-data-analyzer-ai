"""
Blob Converter - Convert files to blob format
Supports images, CSV files, and other binary/text files
"""
import base64
import os
import mimetypes
from typing import Optional, Dict, Any


def file_to_blob(file_path: str) -> Dict[str, Any]:
    """
    Convert any file to blob format
    
    Args:
        file_path: Path to the file to convert
        
    Returns:
        Dictionary containing blob data with metadata
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file cannot be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Get file info
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            # Default MIME types for common extensions
            mime_map = {
                '.csv': 'text/csv',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.txt': 'text/plain',
                '.json': 'application/json'
            }
            mime_type = mime_map.get(file_ext, 'application/octet-stream')
        
        # Read file content
        mode = 'rb' if mime_type.startswith('image/') or 'octet-stream' in mime_type else 'r'
        
        if mode == 'rb':
            # Binary mode for images and binary files
            with open(file_path, 'rb') as file:
                file_content = file.read()
                # Encode binary content to base64
                blob_data = base64.b64encode(file_content).decode('utf-8')
                content_encoding = 'base64'
        else:
            # Text mode for CSV, TXT, JSON etc.
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                # Encode text content to base64 for consistency
                blob_data = base64.b64encode(file_content.encode('utf-8')).decode('utf-8')
                content_encoding = 'base64'
        
        # Create blob object
        blob = {
            'data': blob_data,
            'metadata': {
                'filename': file_name,
                'size': file_size,
                'mime_type': mime_type,
                'extension': file_ext,
                'encoding': content_encoding,
                'original_path': file_path
            }
        }
        
        return blob
        
    except Exception as e:
        raise Exception(f"Error converting file to blob: {e}")


def blob_to_file(blob: Dict[str, Any], output_path: str) -> None:
    """
    Convert blob back to file
    
    Args:
        blob: Blob dictionary with data and metadata
        output_path: Path where to save the file
    """
    try:
        blob_data = blob['data']
        metadata = blob['metadata']
        encoding = metadata.get('encoding', 'base64')
        mime_type = metadata.get('mime_type', 'application/octet-stream')
        
        # Decode base64 data
        if encoding == 'base64':
            file_content = base64.b64decode(blob_data)
        else:
            file_content = blob_data.encode('utf-8')
        
        # Write file
        mode = 'wb' if mime_type.startswith('image/') or 'octet-stream' in mime_type else 'w'
        
        if mode == 'wb':
            with open(output_path, 'wb') as file:
                file.write(file_content)
        else:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(file_content.decode('utf-8'))
        
        print(f"✅ Blob converted back to file: {output_path}")
        
    except Exception as e:
        raise Exception(f"Error converting blob to file: {e}")


def print_blob_info(blob: Dict[str, Any]) -> None:
    """
    Print blob information in a readable format
    
    Args:
        blob: Blob dictionary to display
    """
    metadata = blob['metadata']
    data_preview = blob['data']
    
    print(f"""
📄 BLOB INFORMATION
{'='*50}
📁 Filename: {metadata['filename']}
📏 Size: {metadata['size']} bytes
🎭 MIME Type: {metadata['mime_type']}
📎 Extension: {metadata['extension']}
🔐 Encoding: {metadata['encoding']}
📂 Original Path: {metadata['original_path']}
📊 Data Preview: {data_preview}
📝 Total Data Length: {len(blob['data'])} characters
{'='*50}
""")


if __name__ == "__main__":
    print("🔄 Blob Converter - Converting files to blob format")
    print("=" * 60)
    
    # Files to convert
    files_to_convert = [
        "temp_chart.png",
        "test_data.csv"
    ]
    
    converted_blobs = {}
    
    for file_path in files_to_convert:
        print(f"\n🔄 Converting: {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path} - Creating sample file...")
                
                # Create sample files if they don't exist
                if file_path == "temp_chart.png":
                    # Create a simple plot and save as PNG
                    try:
                        import matplotlib.pyplot as plt
                        import numpy as np
                        
                        # Create sample chart
                        x = np.linspace(0, 10, 100)
                        y = np.sin(x)
                        
                        plt.figure(figsize=(8, 6))
                        plt.plot(x, y, 'b-', linewidth=2)
                        plt.title('Sample Chart for Blob Conversion')
                        plt.xlabel('X axis')
                        plt.ylabel('Y axis')
                        plt.grid(True)
                        plt.savefig(file_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print(f"✅ Created sample chart: {file_path}")
                        
                    except ImportError:
                        print("❌ matplotlib not available, skipping chart creation")
                        continue
                
                elif file_path == "test_data.csv":
                    # Create sample CSV
                    import pandas as pd
                    
                    sample_data = {
                        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                        'age': [25, 30, 35, 28, 32],
                        'salary': [50000, 60000, 75000, 55000, 65000],
                        'department': ['Sales', 'Engineering', 'Engineering', 'Sales', 'Marketing']
                    }
                    
                    df = pd.DataFrame(sample_data)
                    df.to_csv(file_path, index=False)
                    print(f"✅ Created sample CSV: {file_path}")
            
            # Convert to blob
            blob = file_to_blob(file_path)
            converted_blobs[file_path] = blob
            
            print(f"✅ Successfully converted: {file_path}")
            
            # Print blob information
            print_blob_info(blob)
            
        except Exception as e:
            print(f"❌ Error converting {file_path}: {e}")
    
    # Summary
    print(f"\n🎉 Conversion Summary:")
    print(f"📊 Successfully converted {len(converted_blobs)} files to blob format")
    
    if converted_blobs:
        print(f"\n💾 Converted files:")
        for filename, blob in converted_blobs.items():
            size_kb = blob['metadata']['size'] / 1024
            print(f"   • {filename} ({size_kb:.2f} KB) → {len(blob['data'])} chars blob")
