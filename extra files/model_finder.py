import google.generativeai as genai

# PASTE YOUR KEY HERE
genai.configure(api_key="AIzaSyBmG9W167SEjKo02_b5I02LdxVii-eMMK4") 

print("🔍 Scanning for available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ AVAILABLE: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")