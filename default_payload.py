MODELS = [
    "gpt_4_omni_mini_128k",
    "gpt_4_omni_128k",
    "cerebras-llama3.1-70b",
    "claude-3-5-sonnet-20240620",
]
DEFAULT_PAYLOAD = {
    "before_date": None,
    "after_date": None,
    "rag_budget": "default",
    "general_web_search": False,
    "academic_web_search": False,
    "custom_urls": [],
    "user_pdf_documents": [],
    "custom_pdf_urls": [],
    "custom_images": [],
    "search_results_language": None,
    "use_perplexity": True,
    "cot": False,
    # # Image related user options to support
    "image_style": "",
    "image_height": 0,
    "image_width": 0,
    "image_style": "auto",
    # # Blocks related user options to support
    "title": True,
    "paragraphs": True,
    "metrics": True,
    "images": True,
    "ai_images": False,
    "web_graphs": True,
    "ai_graphs": True,
    "quotes": True,
    "tables": True,
    "headers": True,
    "hero_image": True,
    "tweets": False,
    # # General user options to support
    "format": "turbo_article",
    "user_query": "explain string theory",
    "audience": None,
    "response_length": "1 page",
    "response_language": "english",
    "response_model": "gpt_4_omni_mini_128k",
    "personality": None,
    "custom_instructions": None,
    "score_threshold": 5,
    "user_images": [],
    "user_pdf_urls": [],
    "user_pdf_documents": [],
    "user_urls": [],
    "user_pre_processed_sources": [],
}
