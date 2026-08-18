import json
import os
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class SkinSenseRAG:
    def __init__(self, kb_path="data/knowledge_base.json"):
        self.kb_path = kb_path
        self.knowledge_base = self._load_kb()
        
        self.use_openai = OPENAI_AVAILABLE
        
        if self.use_openai:
            try:
                # Point to local Ollama server
                self.client = OpenAI(
                    base_url='http://localhost:11434/v1',
                    api_key='ollama' # Required by the SDK, but ignored by Ollama
                )
                print("Ollama Local API successfully configured for RAG.")
            except Exception as e:
                print(f"Error configuring Ollama API: {e}")
                self.use_openai = False
        
        if not self.use_openai:
            print("Running in OFFLINE mode: Using local knowledge base for RAG.")

    def _load_kb(self):
        if os.path.exists(self.kb_path):
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        return {}

    def retrieve_info(self, disease_name):
        # 1. Try OpenAI dynamic generation if available online
        if self.use_openai:
            try:
                prompt = f"""
                You are a medical AI assistant. Our computer vision model just diagnosed a skin lesion as: {disease_name}.
                Please provide a highly professional, clinical response formatted exactly as a valid JSON object with these keys:
                "description" (a single concise string: a 2-3 sentence clinical overview of the disease)
                "symptoms" (list of strings)
                "recommendations" (list of strings)
                "dos" (list of strings)
                "donts" (list of strings)
                Keep it concise, scientific, and suitable for a medical conference presentation.
                Provide ONLY the valid JSON, no markdown formatting blocks.
                """
                response = self.client.chat.completions.create(
                    model="llama3",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                
                # Clean up response string (sometimes it includes ```json tags)
                clean_text = response.choices[0].message.content.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                    
                dynamic_info = json.loads(clean_text)
                return dynamic_info
                
            except Exception as e:
                print(f"OpenAI API failed or timed out: {e}. Falling back to offline local knowledge base.")
                # Fall through to offline mode

        # 2. Offline Fallback (Guaranteed to work during live presentation)
        return self.knowledge_base.get(disease_name, {
            "description": "A skin condition has been identified. Please consult a certified dermatologist for a detailed clinical assessment.",
            "symptoms": ["Consult specific medical literature for targeted symptoms."],
            "recommendations": ["Clinical correlation recommended.", "Consider excisional biopsy for gold standard diagnosis."],
            "dos": ["Seek professional dermatological review."],
            "donts": ["Do not delay medical evaluation."]
        })

    def chat(self, user_message, context=""):
        if self.use_openai:
            try:
                prompt = f"You are a professional medical AI assistant for dermatological diseases. Use this context if relevant: '{context}'. User asks: '{user_message}'. Answer concisely and clinically. Do not use markdown."
                response = self.client.chat.completions.create(
                    model="llama3",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error connecting to AI: {str(e)}. Please refer to the glossary."
        else:
            return "Offline mode: AI Chat is currently disabled. Please refer to the glossary."
