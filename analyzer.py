"""
Analyzer - Minimal data analysis with Ollama
A lightweight implementation that allows natural language queries on pandas DataFrames
"""
import boto3
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List
import ast
import hashlib
import os
import sys
import subprocess
import builtins


class SafeSandbox:
    """Simple sandbox for safe code execution"""
    
    def __init__(self):
        self.blocked_modules = {
            'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
            'http', 'ftplib', 'smtplib', 'pickle', 'shelve', 'marshal',
            'eval', 'exec', 'compile', '__import__', 'open'
        }
        
        self.blocked_builtins = {
            'eval', 'exec', 'compile', 'open', 'input',
            'raw_input', 'file', 'execfile', 'reload', 'vars', 'locals', 'globals'
        }
    
    def _validate_code(self, code: str) -> bool:
        """Validate code for security issues"""
        # Check for blocked keywords (more precise patterns)
        dangerous_patterns = [
            r'\bos\.',
            r'\bsys\.',
            r'\bsubprocess\.',
            r'\bsocket\.',
            r'\burllib\.',
            r'\brequests\.',
            r'\bhttp\.',
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\bcompile\s*\(',
            r'\bfile\s*\(',
            r'system\s*\(',
            r'popen\s*\(',
            r'import\s+(?:os|sys|subprocess|socket|urllib|requests|http)(?:\s|$)',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        
        # Check AST for dangerous nodes
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.blocked_modules:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.blocked_modules:
                        return False
        except SyntaxError:
            return False
            
        return True
    
    def execute(self, code: str, environment: dict) -> dict:
        """Execute code in safe environment"""
        if not self._validate_code(code):
            raise SecurityError("Code contains potentially dangerous operations")
        
        # Create restricted environment
        safe_env = {
            'pd': environment.get('pd'),
            'plt': environment.get('plt'),
            'sns': environment.get('sns'),
            'result': None,
            '__builtins__': {
                k: v for k, v in builtins.__dict__.items() 
                if k not in self.blocked_builtins
            }
        }
        
        # Add all DataFrames from environment (df, df1, df2, etc.)
        for key, value in environment.items():
            if key.startswith('df') and isinstance(value, pd.DataFrame):
                safe_env[key] = value
        
        # Add numpy if present
        if 'np' in environment:
            safe_env['np'] = environment['np']

        # Add plotly if present
        if 'go' in environment:
            safe_env['go'] = environment['go']
        if 'px' in environment:
            safe_env['px'] = environment['px']
        if 'pio' in environment:
            safe_env['pio'] = environment['pio']

        # Add any extra variables (for agent step context)
        for key, value in environment.items():
            if key.startswith('step_') or key == 'previous_output':
                safe_env[key] = value
        
        try:
            exec(code, safe_env)
            return safe_env.get('result')
        except Exception as e:
            raise ExecutionError(f"Code execution failed: {e}")


class SecurityError(Exception):
    """Raised when code contains security issues"""
    pass


class ExecutionError(Exception):
    """Raised when code execution fails"""
    pass


class OllamaLLM:
    """Simple Ollama LLM wrapper"""
    
    def __init__(self, model: str = "gpt-oss:20b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate(self, prompt: str) -> str:
        """Generate response from Ollama"""
        try:
            import requests
            
            # Create a session with no proxy for localhost
            session = requests.Session()
            session.trust_env = False  # Ignore environment proxy settings
            
            # Explicitly set no proxy for localhost
            session.proxies = {
                'http': None,
                'https': None,
                'no_proxy': 'localhost,127.0.0.1'
            }
            
            response = session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=600  # 10 mins
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")


class LMStudioLLM:
    """LM Studio LLM wrapper using OpenAI-compatible API"""
    
    def __init__(self, model: str = "openai/gpt-oss-20b", base_url: str = "http://localhost:1234", temperature: float = 1, max_tokens: int = -1):
        """
        Initialize LM Studio LLM
        
        Args:
            model: Model identifier (e.g., "openai/gpt-oss-20b")
            base_url: LM Studio server URL (default: http://localhost:1234)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (-1 for unlimited)
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, prompt: str) -> str:
        """
        Generate response from LM Studio using chat completions API
        
        Args:
            prompt: The prompt text to send to the model
            
        Returns:
            Generated response text
        """
        try:
            import requests
            
            # Create a session with no proxy for localhost
            session = requests.Session()
            session.trust_env = False  # Ignore environment proxy settings
            
            # Explicitly set no proxy for localhost
            session.proxies = {
                'http': None,
                'https': None,
                'no_proxy': 'localhost,127.0.0.1'
            }
            
            # Convert single prompt to chat format
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = session.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False
                },
                timeout=600  # 10 mins
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"LM Studio API error: {e}")


class BedrockLLM:
    """AWS Bedrock LLM wrapper for Intel GNAI"""
    
    def __init__(self, model: str = "claude-4-sonnet", base_url: str = "https://gnai.intel.com/api/providers/aws/bedrock", max_tokens: int = 10000, gnai_token: Optional[str] = None):
        """
        Initialize Bedrock LLM
        
        Args:
            model: Model identifier (e.g., "claude-4-sonnet")
            base_url: Bedrock API endpoint URL (default: Intel GNAI Bedrock endpoint)
            max_tokens: Maximum tokens to generate (default: 10000)
            gnai_token: GNAI authentication token (default: reads from GNAI_TOKEN env var)
        """
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        
        # Set up authentication token
        if gnai_token:
            self.gnai_token = gnai_token
        elif 'GNAI_TOKEN' in os.environ:
            self.gnai_token = os.environ['GNAI_TOKEN']
        else:
            raise ValueError("GNAI_TOKEN environment variable must be set or passed as gnai_token parameter")
    
    def generate(self, prompt: str) -> str:
        """
        Generate response from Bedrock using boto3 client
        
        Args:
            prompt: The prompt text to send to the model
            
        Returns:
            Generated response text
        """
        try:
            # Set AWS bearer token fresh at call time
            os.environ['AWS_BEARER_TOKEN_BEDROCK'] = self.gnai_token

            # Create Bedrock client with custom endpoint
            client = boto3.client(
                service_name='bedrock-runtime',
                endpoint_url=self.base_url,
                region_name='gnai'
            )
            
            # Prepare request body in Anthropic format
            request_body = {
                'anthropic_version': 'bedrock-2023-05-31',
                'messages': [{
                    'role': 'user',
                    'content': [{'type': 'text', 'text': prompt}]
                }],
                'max_tokens': self.max_tokens
            }
            
            # Invoke the model
            response = client.invoke_model(
                modelId=self.model,
                body=json.dumps(request_body)
            )
            
            # Parse response
            result = json.loads(response['body'].read())
            
            # Extract text from Anthropic response format
            if 'content' in result and len(result['content']) > 0:
                return result['content'][0]['text']
            else:
                raise Exception(f"Unexpected response format: {result}")
                
        except Exception as e:
            raise Exception(f"Bedrock API error: {e}")


class AnthropicLLM:
    """Anthropic LLM wrapper for Intel GNAI"""
    
    def __init__(self, model: str = "claude-4-5-sonnet", base_url: str = "https://gnai.intel.com/api/providers/anthropic", max_tokens: int = 10000, gnai_token: Optional[str] = None, cert_path: Optional[str] = None):
        """
        Initialize Anthropic LLM
        
        Args:
            model: Model identifier (e.g., "claude-4-5-sonnet")
            base_url: Anthropic API endpoint URL (default: Intel GNAI Anthropic endpoint)
            max_tokens: Maximum tokens to generate (default: 10000)
            gnai_token: GNAI authentication token (default: reads from GNAI_TOKEN env var)
            cert_path: Path to SSL certificate bundle (default: C:\\Users\\mtale\\intel-certs\\intel-ca-bundle.crt)
        """
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        
        # Set up authentication token
        if gnai_token:
            self.auth_token = gnai_token
        elif 'GNAI_TOKEN' in os.environ:
            self.auth_token = os.environ['GNAI_TOKEN']
        else:
            raise ValueError("GNAI_TOKEN environment variable must be set or passed as gnai_token parameter")

        # Set certificate path
        self.cert_path = cert_path or "C:\\Users\\mtale\\intel-certs\\intel-ca-bundle.crt"
        
        # Verify certificate file exists
        if not os.path.exists(self.cert_path):
            print(f"WARNING: Certificate file not found: {self.cert_path}")

    def generate(self, prompt: str) -> str:
        """
        Generate response from Anthropic using httpx client

        Args:
            prompt: The prompt text to send to the model

        Returns:
            Generated response text
        """
        try:
            import httpx
            import ssl
            from anthropic import Anthropic
            
            # Get proxy settings from environment
            proxy = None
            if 'HTTPS_PROXY' in os.environ or 'https_proxy' in os.environ:
                proxy = os.environ.get('HTTPS_PROXY', os.environ.get('https_proxy'))
            
            # Create SSL context with certificate
            ssl_context = ssl.create_default_context()
            ssl_context.load_verify_locations(self.cert_path)
            
            # Create httpx client with SSL and proxy support
            http_client = httpx.Client(
                verify=ssl_context,
                proxy=proxy,
                timeout=30.0
            )
            
            # Create Anthropic client
            client = Anthropic(
                base_url=self.base_url,
                auth_token=self.auth_token,
                http_client=http_client
            )
            
            # Create message
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                raise Exception(f"Unexpected response format: {response}")
                
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")


class DataFrameAnalyzer:
    """Main class for natural language DataFrame queries"""
    
    def __init__(self, dataframes: List[Dict[str, Any]], llm = None, use_sandbox: bool = True, initial_prompt: Optional[str] = None):
        """
        Initialize analyzer with multiple DataFrames
        
        Args:
            dataframes: List of dictionaries with 'dataframe' and 'dataframe_description' keys
                       Example: [{"dataframe": df1, "dataframe_description": "Sales data"}, ...]
            llm: LLM instance (OllamaLLM or LMStudioLLM)
            use_sandbox: Enable sandbox execution
            initial_prompt: Optional custom initial prompt
        """
        self.dataframes = dataframes
        self.llm = llm 
        self.conversation_history: List[Dict[str, str]] = []
        self.use_sandbox = use_sandbox
        self.sandbox = SafeSandbox() if use_sandbox else None
        self.initial_prompt = initial_prompt
    
    def _get_df_info(self) -> str:
        """Get dataframe schema and sample data for all DataFrames"""
        all_tables_info = []
        
        for idx, df_dict in enumerate(self.dataframes):
            df = df_dict.get('dataframe')
            description = df_dict.get('dataframe_description', f'DataFrame {idx}')
            df_var_name = f'df{idx}' if idx > 0 else 'df'
            
            if df is None or not isinstance(df, pd.DataFrame):
                continue
            
            # Calculate a simple hash for table name
            column_string = ",".join(df.columns)
            table_hash = hashlib.md5(column_string.encode()).hexdigest()[:8]
            table_name = f"table_{table_hash}"
            
            # Build column metadata
            columns_info = []
            for col, dtype in df.dtypes.items():
                # Map pandas dtypes to simplified types
                if pd.api.types.is_numeric_dtype(dtype):
                    col_type = "number" if pd.api.types.is_float_dtype(dtype) else "integer"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    col_type = "datetime"
                elif pd.api.types.is_bool_dtype(dtype):
                    col_type = "boolean"
                else:
                    col_type = "text"
                
                columns_info.append({
                    "name": str(col),
                    "type": col_type,
                    "description": None,
                    "expression": None,
                    "alias": None
                })
            
            # Build table info with description
            table_info = f'\n### DataFrame Variable: `{df_var_name}`\n'
            table_info += f'**Description:** {description}\n'
            table_info += f'<table dialect="postgres" table_name="{table_name}" '
            table_info += f'variable_name="{df_var_name}" '
            table_info += f'columns="{json.dumps(columns_info, ensure_ascii=False)}" '
            table_info += f'dimensions="{df.shape[0]}x{df.shape[1]}">\n'
            
            # Add CSV data (first 5 rows)
            # df_head = df.head()
            # table_info += df_head.to_csv(index=False)
            table_info += "</table>\n"
            
            all_tables_info.append(table_info)
        
        return '\n'.join(all_tables_info)
    
    def _build_prompt(self, query: str) -> str:
        """Build prompt structure"""
        df_info = self._get_df_info()
        
        # Agent description (system message)
        system_msg = """You are an AI data analyst that helps with data analysis tasks. You generate Python code to answer questions about pandas DataFrames.
You have access to the following tools and libraries:
- pandas (imported as pd) for data manipulation
- matplotlib.pyplot (imported as plt) for plotting
- seaborn (imported as sns) for statistical visualization
- Any other standard Python libraries as needed

You will have access to multiple DataFrames, each with a variable name (df, df1, df2, etc.) and a description of what data it contains.

Always follow these rules:
1. Generate clean, executable Python code
2. Multiple DataFrames are available as variables (df, df1, df2, etc.) - check the table information below
3. Store your final answer in a variable called 'result' as a dictionary with 'type', 'value', and 'reply' keys
4. For visualizations, use matplotlib/seaborn by default and save as 'temp_chart.png' (type 'plot')
5. Only use Plotly (type 'html') if the user explicitly requests interactive, plotly, or html output
6. Use descriptive variable names and add comments for clarity

You might get data between two or more driver versions of GPU and their improvement you may need to compare them."""

        if self.initial_prompt:
            system_msg += "\nINITIAL USER PROMPT:\n " + self.initial_prompt
        # Previous conversation context
        history_context = ""
        if self.conversation_history:
            history_context = "\n### PREVIOUS CONVERSATION\n"
            for i, entry in enumerate(self.conversation_history[-3:], 1):
                history_context += f"Question: {entry['query']}\n"
                if 'error' in entry:
                    history_context += f"Error in previous attempt: {entry['error']}\n"
                    if entry.get('code'):
                        history_context += f"Failed code:\n```python\n{entry['code']}\n```\n"
                    history_context += "Please fix the error and try again.\n\n"
                elif 'result' in entry:
                    history_context += f"Answer: {entry['result']}\n\n"

        # Last code generated (if any)
        last_code_section = ""
        if self.conversation_history and 'code' in self.conversation_history[-1]:
            last_code_section = f"\nLast code generated:\n```python\n{self.conversation_history[-1]['code']}\n```\n"
        else:
            last_code_section = """
Update this initial code:
```python
# TODO: import the required dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Write code here

# Declare result var with type, value, and reply fields
# Examples: 
# { "type": "string", "value": f"The highest salary is {highest_salary}.", "reply": "The highest salary is $75,000" }
# { "type": "number", "value": 125, "reply": "The total count is 125" }
# { "type": "dataframe", "value": pd.DataFrame({...}), "reply": "Here's the filtered data" }
# { "type": "plot", "value": "temp_chart.png", "reply": "I've created a visualization" }
result = {"type": "...", "value": ..., "reply": "..."}
```
"""

        # Build final prompt
        prompt = f"""{system_msg}

{history_context}

<tables>
{df_info}
</tables>

{last_code_section}

At the end, declare "result" variable as a dictionary with type, value, and reply in the following format:
type (possible values "string", "number", "dataframe", "plot", "html"). Examples: 
- {{ "type": "string", "value": f"The highest salary is {{highest_salary}}.", "reply": "The highest salary is $75,000" }} 
- {{ "type": "number", "value": 125, "reply": "The total count is 125" }} 
- {{ "type": "dataframe", "value": pd.DataFrame({{...}}), "reply": "Here's the filtered data with 5 rows" }} 
- {{ "type": "plot", "value": "temp_chart.png", "reply": "I've created a visualization showing the trend" }}

The "reply" field should contain a human-friendly explanation of the result.

Generate python code and return full updated code:

### Note: Use matplotlib/seaborn for visualizations and save as 'temp_chart.png' with type "plot". Only use Plotly with type "html" (pio.to_html(fig, full_html=False, include_plotlyjs='cdn')) if the user explicitly asks for interactive, plotly, or html output.

# User Query: 
{query}"""

        return prompt
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Try to extract code from markdown blocks
        code_match = re.search(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            # If no markdown block, try to find code-like content
            code = response
        
        # Clean up common issues
        code = code.strip()
        
        # Validate it's valid Python
        try:
            ast.parse(code)
        except SyntaxError:
            raise ValueError("Generated code is not valid Python")
        
        return code
    
    def _execute_code(self, code: str) -> Any:
        """Execute generated code safely with sandbox"""
        # Create execution environment with plotting libraries
        exec_globals = {
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'result': None,
            'os': os,
            'np': None  # Will import numpy if needed
        }

        # Inject plotly if available
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import plotly.io as pio
            exec_globals['go'] = go
            exec_globals['px'] = px
            exec_globals['pio'] = pio
        except ImportError:
            pass
        
        # Inject all DataFrames into the environment
        for idx, df_dict in enumerate(self.dataframes):
            df = df_dict.get('dataframe')
            if df is not None and isinstance(df, pd.DataFrame):
                df_var_name = f'df{idx}' if idx > 0 else 'df'
                exec_globals[df_var_name] = df.copy()  # Work on a copy
        
        # Try to import numpy if code uses it
        if 'np.' in code or 'numpy' in code:
            try:
                import numpy as np
                exec_globals['np'] = np
            except ImportError:
                pass
        
        try:
            # Set matplotlib to non-interactive backend for saving plots
            plt.switch_backend('Agg')
            
            # Execute the code with or without sandbox
            if self.use_sandbox and self.sandbox:
                result = self.sandbox.execute(code, exec_globals)
            else:
                exec(code, exec_globals)
                result = exec_globals.get('result')
            
            if result is None:
                raise ValueError("Code did not set 'result' variable")
            
            # Handle different result formats
            if isinstance(result, dict) and 'type' in result and 'value' in result:
                # Structured result format
                return result
            else:
                # Legacy format - wrap in dictionary
                if isinstance(result, pd.DataFrame):
                    return {"type": "dataframe", "value": result}
                elif isinstance(result, (int, float)):
                    return {"type": "number", "value": result}
                elif isinstance(result, str) and result.endswith('.png'):
                    return {"type": "plot", "value": result}
                elif isinstance(result, str) and '<div' in result:
                    return {"type": "html", "value": result}
                else:
                    return {"type": "string", "value": str(result)}
                    
        except (SecurityError, ExecutionError) as e:
            raise Exception(f"Sandbox error: {e}")
        except Exception as e:
            raise Exception(f"Code execution error: {e}")
        finally:
            # Reset matplotlib backend
            plt.close('all')
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """Format result for display"""
        if not isinstance(result, dict) or 'type' not in result or 'value' not in result:
            return str(result)
        
        result_type = result['type']
        value = result['value']
        
        if result_type == "dataframe":
            if isinstance(value, pd.DataFrame):
                value.to_csv('last_result.csv', index=False)  # Save last result for reference
                return f"\n{value.to_string()}"
            elif isinstance(value, pd.Series):
                return f"\n{value.to_string()}"
            else:
                return str(value)
        elif result_type == "html":
            return f"Interactive Plotly chart generated ({len(value)} bytes of HTML)"
        elif result_type == "plot":
            return f"Plot saved as: {value}"
        elif result_type == "number":
            return str(value)
        elif result_type == "string":
            return str(value)
        else:
            return str(value)
    
    def chat(self, query: str, max_retries: int = 2) -> str:
        """
        Ask a question about the DataFrame in natural language
        
        Args:
            query: Natural language question
            max_retries: Number of retries if code generation fails
            
        Returns:
            Answer as a string
        """
        print(f"\n[History] {self.conversation_history}")
        print("=" * 60)
        print(f"\n[Question] {query}")
        
        for attempt in range(max_retries + 1):
            try:
                # Generate prompt
                prompt = self._build_prompt(query)
                print("Prompt:")
                print(prompt)
                
                # Get code from LLM
                print(f"[LLM] Generating code... (attempt {attempt + 1})")
                response = self.llm.generate(prompt)
                
                # Extract and clean code
                code = self._extract_code(response)
                print(f"\n[Code] Generated code:\n{code}\n")
                
                # Execute code
                print("[Exec] Executing code...")
                result = self._execute_code(code)
                
                # Extract reply from result, or format it if not provided
                if isinstance(result, dict) and 'reply' in result:
                    reply = result['reply']
                else:
                    # Fallback to formatted result if LLM didn't provide reply
                    reply = self._format_result(result)
                
                # Store in conversation history (success case)
                self.conversation_history.append({
                    'query': query,
                    'code': code,
                    'result': reply
                })
                
                print(f"[OK] Answer: {reply}")
                
                # Return in the new format with code, result, and reply
                return {
                    "code": code,
                    "result": result,
                    "reply": reply
                }
                
            except Exception as e:
                print(f"[Warn] Error: {e}")
                
                # Add exception to conversation history for next retry
                self.conversation_history.append({
                    'query': query,
                    'code': code if 'code' in locals() else None,
                    'error': str(e)
                })
                
                if attempt < max_retries:
                    print("[Retry] Retrying...")
                else:
                    error_msg = f"Failed to answer query after {max_retries + 1} attempts: {e}"
                    return {
                        "code": "",
                        "result": {"type": "error", "value": str(e)},
                        "reply": error_msg
                    }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("[Clear] Conversation history cleared")

    # ══════════════════════════════════════════════════════════════════════
    #  AGENTIC MULTI-STEP ANALYSIS
    # ══════════════════════════════════════════════════════════════════════

    def _get_df_metadata(self) -> str:
        """
        Return lightweight metadata for ALL DataFrames -- column names, types,
        shapes, and descriptions -- WITHOUT any actual row data.
        This keeps the prompt small even with hundreds of files.
        """
        lines = []
        for idx, df_dict in enumerate(self.dataframes):
            df = df_dict.get('dataframe')
            description = df_dict.get('dataframe_description', f'DataFrame {idx}')
            var_name = f'df{idx}' if idx > 0 else 'df'

            if df is None or not isinstance(df, pd.DataFrame):
                continue

            col_summary = []
            for col, dtype in df.dtypes.items():
                if pd.api.types.is_numeric_dtype(dtype):
                    col_type = "float" if pd.api.types.is_float_dtype(dtype) else "int"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    col_type = "datetime"
                elif pd.api.types.is_bool_dtype(dtype):
                    col_type = "bool"
                else:
                    col_type = "str"
                col_summary.append(f"  - {col} ({col_type})")

            lines.append(
                f"### `{var_name}` -- {description}\n"
                f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                f"Columns:\n" + "\n".join(col_summary)
            )
        return "\n\n".join(lines)

    def _build_agent_step_prompt(self, query: str, step: int,
                                  previous_output: str,
                                  all_code_so_far: str) -> str:
        """
        Build the prompt for one step of the agent loop.

        * Step 1  : Agent sees only metadata + query.  It should generate code
                    that reads / filters / selects whatever it needs from the
                    available DataFrames.
        * Step 2+ : Agent sees the previous step's output and decides what to
                    do next (further analysis, visualisation, etc.).
        """
        metadata = self._get_df_metadata()

        system = (
            "You are an agentic AI data analyst. You work in multiple steps.\n"
            "Each step you produce Python code that is executed, and you get the\n"
            "output back.  Continue until the user's task is fully done.\n\n"
            "AVAILABLE LIBRARIES:\n"
            "  - pandas (pd), matplotlib.pyplot (plt), seaborn (sns)\n"
            "  - numpy (np)\n"
            "  - plotly.graph_objects (go), plotly.express (px), plotly.io (pio)\n\n"
            "ALL DataFrames are already loaded as variables (df, df1, df2, ...).\n"
            "Do NOT read files from disk -- use the variables directly.\n\n"
            "RULES:\n"
            "1. Always set a `result` dict with keys: type, value, reply, done\n"
            "2. `done` must be True when the task is fully finished, else False\n"
            "3. When `done` is False, `value` should contain intermediate output\n"
            "   (e.g. a summary string or a small DataFrame) that helps the next step.\n"
            "4. Allowed `type` values: 'string', 'number', 'dataframe', 'plot', 'html'\n"
            "5. For interactive charts use plotly: create a fig, then\n"
            "   result = {'type': 'html', 'value': pio.to_html(fig, full_html=False, include_plotlyjs='cdn'), 'reply': '...', 'done': True}\n"
            "6. For static charts use plt, save as 'temp_chart.png',\n"
            "   result = {'type': 'plot', 'value': 'temp_chart.png', 'reply': '...', 'done': True}\n"
            "7. For CSV output use\n"
            "   result = {'type': 'dataframe', 'value': your_df, 'reply': '...', 'done': True}\n"
            "8. Keep each step focused -- do one logical thing per step.\n"
            "9. In early steps, examine and filter data.  In later steps, do analysis / visualization.\n"
        )

        if self.initial_prompt:
            system += f"\nINITIAL CONTEXT FROM USER:\n{self.initial_prompt}\n"

        # Conversation history (compact)
        history = ""
        if self.conversation_history:
            history = "\n### PREVIOUS CONVERSATION (recent)\n"
            for entry in self.conversation_history[-3:]:
                history += f"Q: {entry['query']}\n"
                if 'error' in entry:
                    history += f"Error: {entry['error']}\n"
                elif 'result' in entry:
                    history += f"A: {entry['result']}\n"
                history += "\n"

        step_info = f"\n### CURRENT STEP: {step}\n"
        if step == 1:
            step_info += (
                "This is the FIRST step.  You see only metadata (column names,\n"
                "types, shapes) of the DataFrames below.  Generate code that\n"
                "reads/filters/selects whatever data you need from the DataFrame\n"
                "variables to answer the user's query.  Set done=False if you\n"
                "need more steps, True if you can fully answer now.\n"
            )
        else:
            step_info += (
                f"Previous step output:\n```\n{previous_output}\n```\n\n"
                "Continue the analysis.  You have all original DataFrames plus\n"
                "the output from the previous step shown above.\n"
                "Set done=True when the task is complete.\n"
            )

        code_context = ""
        if all_code_so_far:
            code_context = f"\n### CODE FROM PREVIOUS STEPS (for reference)\n```python\n{all_code_so_far}\n```\n"

        prompt = f"""{system}

{history}

### DATAFRAME METADATA (no row data -- read from variables as needed)
{metadata}

{step_info}
{code_context}

User query: {query}

Generate Python code for this step.  ALWAYS set the `result` variable.
"""
        return prompt

    def chat_agent(self, query: str, max_steps: int = 20, max_retries: int = 2) -> dict:
        """
        Multi-step agentic chat.  The agent iterates up to `max_steps` times,
        each time generating code, executing it, and feeding the output into
        the next step.  Stops early when the agent sets done=True.

        Args:
            query:       Natural language question / task
            max_steps:   Upper bound on agent iterations (default 20)
            max_retries: Retries per step on code-gen failure

        Returns:
            dict with keys: code, result, reply, steps
        """
        print(f"\n{'=' * 60}")
        print(f"[Agent] Query: {query}")
        print(f"[Agent] Max steps: {max_steps}")

        all_code = []
        previous_output = ""
        final_result = None
        final_reply = ""
        steps_taken = 0

        for step in range(1, max_steps + 1):
            steps_taken = step
            print(f"\n--- Agent Step {step} ---")

            step_success = False
            for attempt in range(max_retries + 1):
                try:
                    # Build prompt for this step
                    prompt = self._build_agent_step_prompt(
                        query, step, previous_output, "\n\n".join(all_code)
                    )
                    print(f"[Agent] Step {step} prompt length: {len(prompt)} chars")

                    # Get code from LLM
                    print(f"[LLM] Generating code... (step {step}, attempt {attempt + 1})")
                    response = self.llm.generate(prompt)
                    code = self._extract_code(response)
                    print(f"[Code] Step {step}:\n{code}\n")

                    # Execute code
                    result = self._execute_agent_code(code)
                    all_code.append(f"# --- Step {step} ---\n{code}")

                    # Extract fields
                    done = result.get('done', False)
                    reply = result.get('reply', '')
                    result_type = result.get('type', 'string')
                    value = result.get('value', '')

                    print(f"[Agent] Step {step} done={done} type={result_type}")
                    print(f"[Agent] Reply: {reply}")

                    if done:
                        final_result = result
                        final_reply = reply
                        step_success = True
                        break
                    else:
                        # Prepare intermediate output for next step
                        if isinstance(value, pd.DataFrame):
                            previous_output = (
                                f"DataFrame shape: {value.shape}\n"
                                f"Columns: {list(value.columns)}\n"
                                f"Head:\n{value.head(10).to_string()}"
                            )
                        elif isinstance(value, str) and len(value) > 2000:
                            previous_output = value[:2000] + "\n... (truncated)"
                        else:
                            previous_output = str(value)
                        step_success = True
                        break

                except Exception as e:
                    print(f"[Warn] Step {step} attempt {attempt + 1} error: {e}")
                    if attempt < max_retries:
                        previous_output += f"\n\nERROR in step {step}: {e}\nPlease fix."
                    else:
                        previous_output += f"\n\nFATAL error in step {step}: {e}"

            if final_result is not None:
                break  # Agent signalled done

            if not step_success:
                final_reply = f"Agent failed at step {step} after {max_retries + 1} attempts."
                final_result = {"type": "error", "value": final_reply, "done": True}
                break

        # Fallback if agent never set done=True
        if final_result is None:
            final_reply = (
                f"Agent completed {steps_taken} steps without marking done. "
                f"Last output:\n{previous_output}"
            )
            final_result = {"type": "string", "value": final_reply, "done": True}

        # Record in conversation history
        self.conversation_history.append({
            'query': query,
            'code': "\n\n".join(all_code),
            'result': final_reply
        })

        combined_code = "\n\n".join(all_code)
        print(f"[Agent] Finished in {steps_taken} steps")

        return {
            "code": combined_code,
            "result": final_result,
            "reply": final_reply,
            "steps": steps_taken
        }

    def _execute_agent_code(self, code: str) -> dict:
        """
        Execute code for one agent step.  Similar to _execute_code but adds
        plotly to the environment and handles the `done` flag.
        """
        exec_globals = {
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'result': None,
        }

        # Import plotly
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import plotly.io as pio
            exec_globals['go'] = go
            exec_globals['px'] = px
            exec_globals['pio'] = pio
        except ImportError:
            pass

        # Import numpy
        try:
            import numpy as np
            exec_globals['np'] = np
        except ImportError:
            pass

        # Inject DataFrames
        for idx, df_dict in enumerate(self.dataframes):
            df = df_dict.get('dataframe')
            if df is not None and isinstance(df, pd.DataFrame):
                var = f'df{idx}' if idx > 0 else 'df'
                exec_globals[var] = df.copy()

        try:
            plt.switch_backend('Agg')

            if self.use_sandbox and self.sandbox:
                result = self.sandbox.execute(code, exec_globals)
            else:
                exec(code, exec_globals)
                result = exec_globals.get('result')

            if result is None:
                raise ValueError("Code did not set 'result' variable")

            # Normalise result to dict with expected keys
            if isinstance(result, dict) and 'type' in result:
                result.setdefault('done', True)
                result.setdefault('reply', '')
                result.setdefault('value', '')
                return result

            # Legacy format
            if isinstance(result, pd.DataFrame):
                return {"type": "dataframe", "value": result, "done": True, "reply": ""}
            elif isinstance(result, (int, float)):
                return {"type": "number", "value": result, "done": True, "reply": ""}
            elif isinstance(result, str) and result.endswith('.png'):
                return {"type": "plot", "value": result, "done": True, "reply": ""}
            else:
                return {"type": "string", "value": str(result), "done": True, "reply": ""}

        except (SecurityError, ExecutionError) as e:
            raise Exception(f"Sandbox error: {e}")
        except Exception as e:
            raise Exception(f"Code execution error: {e}")
        finally:
            plt.close('all')


# Convenience function
def create_analyzer(dataframes: List[Dict[str, Any]], model: str = "claude-4-5-sonnet", use_sandbox: bool = True, initial_prompt: Optional[str] = None, llm_provider: str = "anthropic", base_url: Optional[str] = None, gnai_token: Optional[str] = None, cert_path: Optional[str] = None) -> DataFrameAnalyzer:
    """
    Create a DataFrame analyzer with specified LLM
    
    Args:
        dataframes: List of dictionaries with 'dataframe' and 'dataframe_description' keys
                   Example: [{"dataframe": df1, "dataframe_description": "Sales data"}, ...]
        model: Model name (default: claude-4-5-sonnet)
               - For Anthropic: "claude-4-5-sonnet", "claude-3-sonnet", etc.
               - For Bedrock: "claude-4-sonnet", "claude-3-sonnet", etc.
               - For Ollama: "gpt-oss:20b", "llama2", etc.
               - For LM Studio: "openai/gpt-oss-20b", etc.
        use_sandbox: Enable secure sandbox execution (default: True)
        initial_prompt: Optional custom initial prompt for the AI (default: None)
        llm_provider: LLM provider - "anthropic", "bedrock", "ollama", or "lmstudio" (default: "anthropic")
        base_url: Optional custom base URL for the LLM server
                 - Anthropic default: https://gnai.intel.com/api/providers/anthropic
                 - Bedrock default: https://gnai.intel.com/api/providers/aws/bedrock
                 - Ollama default: http://localhost:11434
                 - LM Studio default: http://localhost:1234
        gnai_token: GNAI authentication token (required for Anthropic/Bedrock, optional if GNAI_TOKEN env var is set)
        cert_path: SSL certificate path (for Anthropic, default: C:\\Users\\mtale\\intel-certs\\intel-ca-bundle.crt)
        
    Returns:
        DataFrameAnalyzer instance
    """
    if llm_provider.lower() == "lmstudio":
        url = base_url or "http://localhost:1234"
        llm = LMStudioLLM(model=model, base_url=url)
    elif llm_provider.lower() == "bedrock":
        url = base_url or "https://gnai.intel.com/api/providers/aws/bedrock"
        llm = BedrockLLM(model=model, base_url=url, gnai_token=gnai_token)
    elif llm_provider.lower() == "ollama":
        url = base_url or "http://localhost:11434"
        llm = OllamaLLM(model=model, base_url=url)
    else:  # Default to anthropic
        url = base_url or "https://gnai.intel.com/api/providers/anthropic"
        llm = AnthropicLLM(model=model, base_url=url, gnai_token=gnai_token, cert_path=cert_path)
    
    return DataFrameAnalyzer(dataframes, llm, use_sandbox, initial_prompt)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Analyzer with Ollama - Enhanced Example")
    print("=" * 60)
    
    # Create sample data - multiple DataFrames
    data1 = {
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 75000, 55000, 65000],
        'department': ['Sales', 'Engineering', 'Engineering', 'Sales', 'Marketing']
    }
    df1 = pd.DataFrame(data1)
    
    data2 = {
        'department': ['Sales', 'Engineering', 'Marketing'],
        'budget': [100000, 250000, 80000],
        'headcount': [10, 25, 8]
    }
    df2 = pd.DataFrame(data2)
    
    print("\n[Data] Sample DataFrames:")
    print("\ndf - Employee Data:")
    print(df1)
    print("\ndf1 - Department Budget Data:")
    print(df2)
    
    # Create analyzer with multiple DataFrames
    dataframes = [
        {
            "dataframe": df1,
            "dataframe_description": "Employee information including name, age, salary, and department"
        },
        {
            "dataframe": df2,
            "dataframe_description": "Department budget and headcount information"
        }
    ]
    
    analyzer = create_analyzer(dataframes)
    
    # Ask questions that might use multiple DataFrames
    analyzer.chat("What is the average salary?")
    analyzer.chat("Which department has the highest budget per employee?")
    analyzer.chat("Show me employees in departments with budget over 100000")
