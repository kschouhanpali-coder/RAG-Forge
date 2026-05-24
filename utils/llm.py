import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import GEMINI_MODEL, DEFAULT_TEMPERATURE, PROMPT_TEMPLATE

class GeminiLLM:
    def __init__(self, api_key: str = None, temperature: float = DEFAULT_TEMPERATURE):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.temperature = temperature
        self.llm = None
        
        self.model_name = GEMINI_MODEL
        if self.api_key:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
                timeout=15.0,
                max_retries=3
            )
            
        # Set up Prompt Template
        self._loaded_prompt_template_str = PROMPT_TEMPLATE
        self.prompt_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

    def is_configured(self) -> bool:
        """Checks if the LLM has been successfully configured with an API key."""
        llm_mode = st.session_state.get("llm_mode", "Gemini API")
        if llm_mode == "Mock LLM (Offline / No Key)":
            return True
        return self.llm is not None

    def clean_mock_response(self, text: str) -> str:
        """
        Cleans and refines the mock response text to ensure it looks professional,
        properly formatted, and free of page headers, duplicate terms, or raw metadata.
        """
        if not text:
            return ""
            
        import re
        # 1. Remove common metadata/headers phrases
        metadata_patterns = [
            r'\btest\s+document\b',
            r'\bdocument\b',
            r'\bpage\s+\d+\b',
            r'\bchapter\s+\d+\b',
            r'\bsection\s+\d+\b',
            r'\bheader\b',
            r'\bfooter\b'
        ]
        for pattern in metadata_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
        # 2. If there's a question mark (e.g. "What is RAG? Retrieval-Augmented..."), strip the question part
        if "?" in text:
            parts = text.split("?", 1)
            text = parts[1].strip()
            
        # 3. Remove duplicate duplicate phrases (e.g. repeating "Retrieval-Augmented Generation (RAG)")
        phrases_to_deduplicate = [
            "Retrieval-Augmented Generation (RAG)",
            "Retrieval-Augmented Generation",
            "RAG"
        ]
        for phrase in phrases_to_deduplicate:
            if text.lower().count(phrase.lower()) > 1:
                # Find the first index of the phrase case-insensitively
                match = re.search(re.escape(phrase), text, re.IGNORECASE)
                if match:
                    # Keep everything from the first match onwards, and remove other occurrences of this phrase
                    start_idx = match.start()
                    remainder = text[match.end():]
                    # Remove case-insensitive occurrences from the remainder
                    remainder = re.sub(re.escape(phrase), '', remainder, flags=re.IGNORECASE)
                    text = text[start_idx:match.end()] + remainder
                    
        # 4. Clean up punctuation, spaces, and formatting artifacts
        text = re.sub(r'\s+', ' ', text) # normalize whitespace
        text = re.sub(r'\s+([.,!?])', r'\1', text) # fix spaces before punctuation
        text = re.sub(r'\.+', '.', text) # remove multiple periods
        text = text.strip()
        
        # 5. Ensure it starts with a capital letter
        if text:
            text = text[0].upper() + text[1:]
            
        # 6. Ensure it ends with a period
        if text and not text[-1] in ".!?":
            text += "."
            
        return text


    def generate_mock_answer(self, context: str, question: str) -> str:
        """
        Generates a simulated LLM answer locally using retrieved context.
        This provides offline testing capability and bypasses API quota limits.
        Ensures the response strictly adheres to RAG constraints:
        - Single, well-structured paragraph
        - No lists, bullet points, or numbering
        - No multiple paragraphs/newlines
        """
        if not context or "⚠️ It looks like there are no documents" in context:
            return "I don't know the answer because no reference context was retrieved."
            
        # Split context into lines
        import re
        lines = context.splitlines()
        cleaned_sentences = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, separators, and source markers
            if not line or line.startswith("---") or line.endswith("---") or "[Source:" in line:
                continue
                
            # Strip list indicators: numbers (e.g. 1., 2a.), dashes, asterisks, bullets
            line_cleaned = re.sub(r'^(?:(?:\d+[\.\)]?)|[\-\*\u2022\u25E6\u25AA\u25FE])\s*', '', line)
            line_cleaned = line_cleaned.strip()
            
            # Split line into sentences
            sentences = re.split(r'(?<=[.!?])\s+', line_cleaned)
            for s in sentences:
                s_clean = s.strip()
                # Skip headers or lines that are too short (less than 15 chars) and don't end in punctuation
                if not s_clean:
                    continue
                if len(s_clean) < 15 and not s_clean[-1] in ".!?":
                    continue
                # Strip any leftover internal lists/dashes
                s_clean = re.sub(r'\s+[\-\*\u2022]\s+', ' ', s_clean)
                cleaned_sentences.append(s_clean)
            
        if not cleaned_sentences:
            return "I don't know the answer because the retrieved context is empty or formatted incorrectly."
            
        # Extract keywords from the question (excluding common question stop words)
        stop_words = {
            "what", "how", "why", "who", "where", "when", "which", "is", "are", "was", "were", 
            "do", "does", "did", "the", "a", "an", "and", "or", "of", "in", "to", "for", "with", 
            "on", "at", "by", "from", "about", "define", "definition", "meaning", "explain"
        }
        q_words = {w for w in re.findall(r'\b\w{3,}\b', question.lower()) if w not in stop_words}
        
        # Analyze keyword matches in the context
        context_lower = context.lower()
        matched_keywords = {w for w in q_words if w in context_lower}
        missing_keywords = q_words - matched_keywords
        
        # Find sentences containing words from the question to make the mock response look relevant
        scored_sentences = []
        for s in cleaned_sentences:
            s_lower = s.lower()
            score = sum(1 for w in q_words if w in s_lower)
            scored_sentences.append((score, s))
            
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        selected = [s for score, s in scored_sentences[:3]]
        
        if not selected:
            selected = cleaned_sentences[:2]
            
        # Merge selected sentences into a single paragraph
        paragraph = " ".join(selected)
        
        # Clean the text using the advanced professional cleaner
        paragraph = self.clean_mock_response(paragraph)
        
        # If the question contains keywords, but NONE of them are matched anywhere in the context:
        if q_words and not matched_keywords:
            return "Based on the retrieved documents, the context does not contain any information related to your query."
            
        # If there are key missing terms (e.g. asking for "pipeline" of "RAG", but "pipeline" is missing),
        # return a helpful response stating what was found and what is missing.
        if missing_keywords and matched_keywords:
            matched_str = ", ".join([f"'{w}'" for w in sorted(matched_keywords)])
            missing_str = " or ".join([f"'{w}'" for w in sorted(missing_keywords)])
            
            if paragraph and paragraph[0].isupper() and not paragraph[1].isupper():
                paragraph = paragraph[0].lower() + paragraph[1:]
            return f"While the retrieved documents contain information about {matched_str}, they do not mention {missing_str}. For context, the documents state that {paragraph}"
            
        # Otherwise, return a normal professional RAG response
        intro = "Based on the retrieved documents, "
        if paragraph and paragraph[0].isupper() and not paragraph[1].isupper():
            paragraph = paragraph[0].lower() + paragraph[1:]
            
        return f"{intro}{paragraph}"

    def generate_answer(self, context: str, question: str) -> str:
        """
        Generates an answer based on the context and the question using Gemini LLM.
        """
        llm_mode = st.session_state.get("llm_mode", "Gemini API")
        if llm_mode == "Mock LLM (Offline / No Key)":
            return self.generate_mock_answer(context, question)

        if not self.is_configured():
            raise ValueError("Gemini API key is not configured. Please set the GOOGLE_API_KEY in the sidebar or env file.")
            
        # Check if the model name or prompt template has changed in the config (hot-reload support)
        from config import GEMINI_MODEL, PROMPT_TEMPLATE
        if getattr(self, '_loaded_prompt_template_str', None) != PROMPT_TEMPLATE:
            self._loaded_prompt_template_str = PROMPT_TEMPLATE
            self.prompt_template = PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
            
        current_model = getattr(self, 'model_name', None)
        if current_model != GEMINI_MODEL:
            self.model_name = GEMINI_MODEL
            if self.api_key:
                self.llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=self.temperature,
                    timeout=15.0,
                    max_retries=3
                )
            
        # Build prompt using LangChain LCEL or simple formatting
        chain = self.prompt_template | self.llm
        
        # We will try a sequence of model names if we hit quota limits or transient failures.
        primary_model = self.model_name
        fallback_models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-latest"]
        
        # Remove primary model from fallbacks to avoid duplicate attempts
        fallback_models = [m for m in fallback_models if m != primary_model]
        models_to_try = [primary_model] + fallback_models
        
        original_error = None
        last_error = None
        
        for model in models_to_try:
            if model == self.model_name and self.llm is not None:
                current_llm = self.llm
            else:
                try:
                    current_llm = ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=self.api_key,
                        temperature=self.temperature,
                        timeout=15.0,
                        max_retries=3
                    )
                except Exception as init_e:
                    last_error = init_e
                    if original_error is None:
                        original_error = init_e
                    continue
            
            chain = self.prompt_template | current_llm
            
            try:
                response = chain.invoke({"context": context, "question": question})
                
                # If we succeeded with a fallback model, update the instance configuration
                if model != self.model_name:
                    self.model_name = model
                    self.llm = current_llm
                    try:
                        import config
                        config.GEMINI_MODEL = model
                    except Exception:
                        pass
                
                # Clear any active warnings on successful API call
                if "llm_api_warning" in st.session_state:
                    del st.session_state.llm_api_warning
                return response.content
            except Exception as e:
                err_msg = str(e)
                last_error = e
                if original_error is None:
                    original_error = e
                
                # If the error is fatal (invalid key or location not supported), do not try other models
                is_invalid_key = "API_KEY_INVALID" in err_msg or "key is invalid" in err_msg.lower() or "api key not valid" in err_msg.lower()
                is_unsupported_location = "location" in err_msg.lower() and "not supported" in err_msg.lower()
                
                if is_invalid_key or is_unsupported_location:
                    break
                
                # Otherwise, if it is a quota error or a 404 not found, we can try the next candidate model
                is_quota_error = "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "quota" in err_msg.lower()
                is_not_found = "NOT_FOUND" in err_msg or "404" in err_msg or "not found" in err_msg.lower()
                
                if not (is_quota_error or is_not_found):
                    # For other unexpected errors, break to raise it
                    break

        # If all attempts failed, fall back to the mock LLM locally and register a warning message in session state
        orig_msg = str(original_error) if original_error else str(last_error)
        
        if "RESOURCE_EXHAUSTED" in orig_msg or "429" in orig_msg or "quota" in orig_msg.lower():
            warning_msg = (
                "⚠️ **API Quota Exceeded (429 Rate Limit)**: You have exceeded the free tier request limit for Gemini. "
                "The studio is temporarily simulating responses using local Mock LLM mode. Please set a new API key in the sidebar."
            )
        elif "API_KEY_INVALID" in orig_msg or "key is invalid" in orig_msg.lower() or "api key not valid" in orig_msg.lower():
            warning_msg = (
                "⚠️ **Invalid API Key**: The provided Google Gemini API key is incorrect or unauthorized. "
                "The studio is temporarily simulating responses using local Mock LLM mode. Please check your key in the sidebar."
            )
        elif "location" in orig_msg.lower() and "not supported" in orig_msg.lower():
            warning_msg = (
                "⚠️ **User Location Not Supported**: Google Gemini API is not supported in your current geographic location. "
                "The studio is temporarily simulating responses using local Mock LLM mode. Please use a VPN or set up billing in Google AI Studio."
            )
        else:
            warning_msg = f"⚠️ **Gemini Error ({orig_msg})**: The studio is temporarily simulating responses using local Mock LLM mode."
            
        st.session_state.llm_api_warning = warning_msg
        return self.generate_mock_answer(context, question)

