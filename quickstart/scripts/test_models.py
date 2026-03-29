import requests
import json
import time

# Optimized list for local hardware
MODELS = ["phi3.5", "llama3.1", "mistral-nemo", "gemma2"]
ENDPOINT = "http://localhost:11434/api/generate"

def test_model(model_name):
    print(f"\n🔍 Testing Model: {model_name}...")
    payload = {
        "model": model_name,
        "prompt": "Hello! Explain why RAG is better than long-context in 1 sentence.",
        "stream": False
    }
    
    start_time = time.time()
    try:
        # 180s timeout for the very first load of a model
        response = requests.post(ENDPOINT, json=payload, timeout=180)
        response.raise_for_status()
        duration = time.time() - start_time
        answer = response.json().get("response", "").strip()
        print(f"✅ Success! ({duration:.2f}s)")
        print(f"💬 Answer: {answer[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        if "timeout" in str(e).lower():
            print("💡 Tip: This usually means the model is still loading into memory. Try running the script again.")
        return False

def run_diagnostics():
    print("======================================================")
    print("🛡️ RAG FRAMEWORK: MODEL DIAGNOSTICS")
    print("======================================================")
    
    results = {}
    for model in MODELS:
        results[model] = test_model(model)
        
    print("\n" + "="*50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*50)
    for model, success in results.items():
        status = "READY" if success else "NOT FOUND/FAIL"
        print(f" - {model:15}: {status}")
    
    if all(results.values()):
        print("\n🚀 All models are primed! Run: python run_eval.py --local --models phi3.5 llama3.1 mistral-nemo gemma2")
    else:
        print("\n⚠️ Some models failed. Run 'sh quickstart/setup.sh' to get the optimized versions.")

if __name__ == "__main__":
    run_diagnostics()
