import sys
from pathlib import Path
sys.path.insert(0, '.')
import json
import urllib.request
from nids.policy_agent import PolicyAgent

print("Checking Ollama / Qwen status on http://127.0.0.1:11434/api/tags...")
try:
    req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
    with urllib.request.urlopen(req, timeout=3) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        models = [m['name'] for m in data.get('models', [])]
        print(f"[+] Ollama server is ONLINE! Available models: {models}")
except Exception as e:
    print(f"[-] Ollama connection test: {e}")

agent = PolicyAgent()
print(f"[+] PolicyAgent configured with model='{agent.llm_model}', url='{agent.ollama_url}', use_gemini={agent.use_gemini}")

