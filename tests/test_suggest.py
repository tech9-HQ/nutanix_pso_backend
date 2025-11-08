# test_ai_estimation.py
# Run this to test AI estimation independently

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("AI ESTIMATION ENVIRONMENT CHECK")
print("=" * 60)

# Check environment variables
print("\n1. ENVIRONMENT VARIABLES:")
print("-" * 60)

serper_key = os.getenv("SERPER_API_KEY", "")
azure_endpoint = os.getenv("AZURE_ENDPOINT", "")
azure_deployment = os.getenv("AZURE_DEPLOYMENT", "")
azure_key = os.getenv("AZURE_API_KEY", "")
gemini_key = os.getenv("GEMINI_API_KEY", "")
search_provider = os.getenv("SEARCH_PROVIDER", "")
llm_provider = os.getenv("LLM_PROVIDER", "")

print(f"SERPER_API_KEY:    {'✅ Set' if serper_key else '❌ Missing'} ({len(serper_key) if serper_key else 0} chars)")
print(f"AZURE_ENDPOINT:    {'✅ Set' if azure_endpoint else '❌ Missing'}")
print(f"  Value: {azure_endpoint}")
print(f"AZURE_DEPLOYMENT:  {'✅ Set' if azure_deployment else '❌ Missing'} ({azure_deployment})")
print(f"AZURE_API_KEY:     {'✅ Set' if azure_key else '❌ Missing'} ({len(azure_key) if azure_key else 0} chars)")
print(f"GEMINI_API_KEY:    {'✅ Set' if gemini_key else '❌ Missing'} ({len(gemini_key) if gemini_key else 0} chars)")
print(f"SEARCH_PROVIDER:   {search_provider or '❌ Not set'}")
print(f"LLM_PROVIDER:      {llm_provider or '❌ Not set'}")

# Test imports
print("\n2. TESTING IMPORTS:")
print("-" * 60)

try:
    from app.services.duration_estimator import estimate_days_from_web, call_llm
    print("✅ duration_estimator imported successfully")
except Exception as e:
    print(f"❌ Failed to import duration_estimator: {e}")
    sys.exit(1)

# Test web search
print("\n3. TESTING WEB SEARCH:")
print("-" * 60)

if not serper_key:
    print("❌ SERPER_API_KEY not set - skipping search test")
else:
    try:
        import requests
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
        body = {"q": "Nutanix FitCheck duration days", "num": 3}
        
        print("Sending test search request...")
        r = requests.post(url, headers=headers, json=body, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Search successful! Got response")
            
            # Count results
            organic = data.get("organic", [])
            print(f"   Found {len(organic)} organic results")
            
            if organic:
                print(f"   First result: {organic[0].get('title', 'N/A')[:60]}")
        else:
            print(f"❌ Search failed with status {r.status_code}")
            print(f"   Response: {r.text[:200]}")
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")

# Test LLM call
print("\n4. TESTING LLM CALL:")
print("-" * 60)

if not (azure_endpoint and azure_deployment and azure_key):
    print("❌ Azure configuration incomplete - skipping LLM test")
else:
    try:
        print(f"Testing Azure endpoint: {azure_endpoint}")
        print(f"Using deployment: {azure_deployment}")
        
        prompt = "Return only the number 5. No other text."
        
        print("\nSending test LLM request...")
        result = call_llm(prompt, max_tokens=10, temperature=0.0)
        
        if result:
            print(f"✅ LLM call successful!")
            print(f"   Response: {result[:100]}")
        else:
            print("❌ LLM returned None")
            
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        import traceback
        traceback.print_exc()

# Test full AI estimation
print("\n5. TESTING FULL AI ESTIMATION:")
print("-" * 60)

try:
    print("Testing estimate_days_from_web()...")
    
    result = estimate_days_from_web(
        task_text="NCI Cluster FitCheck - Starter on ahv",
        industry="Government",
        deployment_type="on prem",
        source_platform="vmware",
        target_platform="ahv",
        vm_count=50,
        node_count=4,
        search_hints=["Nutanix FitCheck assessment duration"],
    )
    
    if result is not None:
        print(f"✅ AI estimation successful!")
        print(f"   Estimated days: {result}")
    else:
        print("❌ AI estimation returned None")
        print("   Check the logs above for errors")
        
except Exception as e:
    print(f"❌ AI estimation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)

# Provide recommendations
print("\nRECOMMENDATIONS:")
print("-" * 60)

issues = []

if not serper_key:
    issues.append("- Add SERPER_API_KEY to .env file")
if not azure_endpoint or "/models" in azure_endpoint:
    issues.append("- Fix AZURE_ENDPOINT (should not end with /models)")
if not azure_deployment:
    issues.append("- Add AZURE_DEPLOYMENT to .env file")
if not azure_key:
    issues.append("- Add AZURE_API_KEY to .env file")
if search_provider != "serper":
    issues.append("- Set SEARCH_PROVIDER=serper in .env file")
if llm_provider != "azure":
    issues.append("- Set LLM_PROVIDER=azure in .env file")

if issues:
    print("\n".join(issues))
else:
    print("✅ All environment variables look good!")
    print("\nIf AI estimation is still failing, check:")
    print("1. Azure API key is valid and has credits")
    print("2. Serper API key is valid and has credits")
    print("3. Network/firewall allows outbound HTTPS")
    print("4. Check server logs for specific error messages")

print("\n")