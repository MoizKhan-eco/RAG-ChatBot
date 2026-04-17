"""
Test script to verify Organization's OpenAI-compatible API integration via Langfuse
Run this after setting up your .env file with Langfuse credentials
"""

import os
import sys
import requests
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OrgAPIClient:
    """OpenAI-compatible API client for organization's hosted models"""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        chat_model: str = "baseten/zai-org/glm-5",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.base_url = base_url.rstrip("/")
        self.api_url = self.base_url  # Already includes /v1
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
    
    def test_connection(self) -> bool:
        """Test basic connectivity to the API"""
        try:
            # Try to list models (common OpenAI-compatible endpoint)
            response = self.session.get(
                f"{self.api_url}/models",
                timeout=30
            )
            if response.status_code == 200:
                return True
            # Some endpoints don't have /models, try a simple chat
            return self.test_chat() is not None
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def list_models(self) -> Optional[List[str]]:
        """List available models"""
        try:
            response = self.session.get(
                f"{self.api_url}/models",
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                models = [m.get("id", str(m)) for m in data.get("data", [])]
                return models
            return None
        except Exception as e:
            print(f"Failed to list models: {e}")
            return None
    
    def test_chat(self, message: str = "Hello, respond with 'DigiPal API is ready!'") -> Optional[str]:
        """Test chat completion endpoint"""
        try:
            response = self.session.post(
                f"{self.api_url}/chat/completions",
                json={
                    "model": self.chat_model,
                    "messages": [
                        {"role": "user", "content": message}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 100
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            # OpenAI format: choices[0].message.content
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return answer
            
        except requests.exceptions.HTTPError as e:
            print(f"Chat API HTTP error: {e}")
            print(f"Response: {e.response.text if e.response else 'No response'}")
            return None
        except Exception as e:
            print(f"Chat API failed: {e}")
            return None
    
    def test_embeddings(self, texts: List[str] = None) -> Optional[List[List[float]]]:
        """Test embeddings endpoint"""
        if texts is None:
            texts = ["Hello world", "Test embedding"]
        
        try:
            response = self.session.post(
                f"{self.api_url}/embeddings",
                json={
                    "model": self.embedding_model,
                    "input": texts
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            # OpenAI format: data[].embedding
            embeddings = [item["embedding"] for item in data.get("data", [])]
            return embeddings
            
        except requests.exceptions.HTTPError as e:
            print(f"Embeddings API HTTP error: {e}")
            print(f"Response: {e.response.text if e.response else 'No response'}")
            return None
        except Exception as e:
            print(f"Embeddings API failed: {e}")
            return None


def test_env_setup():
    """Test if environment variables are properly configured"""
    required_vars = {
        "API_BASE_URL": os.getenv("API_BASE_URL"),
        "API_KEY": os.getenv("API_KEY"),
        "API_CHAT_MODEL": os.getenv("API_CHAT_MODEL"),
        "API_EMBEDDING_MODEL": os.getenv("API_EMBEDDING_MODEL"),
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    
    print("✅ Environment variables are properly configured")
    print(f"   Base URL: {required_vars['API_BASE_URL']}")
    print(f"   Chat Model: {required_vars['API_CHAT_MODEL']}")
    print(f"   Embedding Model: {required_vars['API_EMBEDDING_MODEL']}")
    return True


def test_api_connection():
    """Test connection to organization's API"""
    client = OrgAPIClient(
        base_url=os.getenv("API_BASE_URL"),
        api_key=os.getenv("API_KEY"),
        chat_model=os.getenv("API_CHAT_MODEL", "baseten/zai-org/glm-5"),
        embedding_model=os.getenv("API_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    
    # Test listing models (optional - may not be supported)
    print("\n📋 Checking available models...")
    models = client.list_models()
    if models:
        print(f"✅ Available models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
    else:
        print("⚠️  Could not list models (this is okay, endpoint may not be supported)")
    
    # Test chat completion
    print(f"\n💬 Testing chat completion (model: {client.chat_model})...")
    chat_response = client.test_chat()
    if chat_response:
        print(f"✅ Chat API working: {chat_response[:100]}...")
    else:
        print("❌ Chat API failed")
        return False
    
    # Test embeddings
    print(f"\n🔢 Testing embeddings (model: {client.embedding_model})...")
    embeddings = client.test_embeddings()
    if embeddings:
        print(f"✅ Embeddings API working: Got {len(embeddings)} embeddings, dimension={len(embeddings[0])}")
    else:
        print("⚠️  Embeddings API failed (may need different model or not supported)")
        print("   Will need to use alternative embedding solution")
    
    return True


if __name__ == "__main__":
    print("🔍 Testing DigiPal Organization API Integration...")
    print("=" * 60)
    
    # Test environment setup
    if not test_env_setup():
        print("\n💡 Please update your .env file with:")
        print("   API_BASE_URL=https://aiapidev.3ecompany.com/v1")
        print("   API_KEY=your-api-key")
        print("   API_CHAT_MODEL=baseten/zai-org/glm-5")
        print("   API_EMBEDDING_MODEL=text-embedding-3-small")
        sys.exit(1)
    
    # Test API connection
    if not test_api_connection():
        print("\n❌ API connection test failed")
        print("\n💡 Check that:")
        print("   1. API_BASE_URL is correct")
        print("   2. API_KEY is valid")
        print("   3. You have access to the specified models")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 API tests completed! Ready to integrate into DigiPal")
    print("\nNext steps:")
    print("1. If all tests passed, update chat.py to use the org API")
    print("2. Run: python chat.py")
