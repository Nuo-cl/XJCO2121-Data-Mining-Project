"""
Utility functions for generating prompts for the DeepSeek emotion classification model.
This module contains three prompt templates with varying amounts of context and explanations.
"""

# Minimal prompt with minimal context
def build_minimal_prompt(batch_texts, examples, emotion_dict):
    """Build minimal prompt with only essential instructions"""
    # Simplified emotion mapping with just the numbers
    emotion_mapping = "EMOTIONS:\n"
    for emotion, idx in emotion_dict.items():
        emotion_mapping += f"{idx}: {emotion}\n"
    
    # Limited examples (max 5)
    examples_text = "EXAMPLES:\n"
    for ex in examples[:min(5, len(examples))]:
        examples_text += f"{ex['text']} -> {ex['label']}\n"
    
    batch_data = "CLASSIFY:\n"
    for i, text in enumerate(batch_texts):
        batch_data += f"{i+1}. {text}\n"
    
    instruction = """Classify each text with the appropriate emotion label number.
Return a JSON object with texts as keys and emotion numbers as values.
Mark your response with %%% at start and end."""
    
    prompt = f"{instruction}\n\n{emotion_mapping}\n\n{examples_text}\n\n{batch_data}\n\nFormat: %%% {{\"text\": emotion_number, ...}} %%%"
    
    return prompt

# Basic prompt template (current standard version)
def build_basic_prompt(batch_texts, examples, emotion_dict):
    """Build basic prompt to send to DeepSeek API (standard version)"""
    emotion_mapping = "Emotion mapping table:\n"
    for emotion, idx in emotion_dict.items():
        emotion_mapping += f"{emotion}: {idx}\n"
    
    examples_text = "Here are example data and their corresponding emotion labels:\n"
    for ex in examples:
        examples_text += f"Text: {ex['text']}\n"
        examples_text += f"Label: {ex['label']}\n"
        examples_text += "---\n"
    
    batch_data = "Please classify the following texts with their corresponding emotion labels:\n"
    for i, text in enumerate(batch_texts):
        batch_data += f"{i+1}. {text}\n"
    
    instruction = """You are a text emotion classification assistant. I will provide some texts, and you'll classify each text's emotion.
Please return the classification results in JSON format where text content is the key and the emotion label number is the value.
Example format: {"text1": 5, "text2": 10, "text3": 27}

In your response, the JSON classification result must be marked with %%% at the beginning and end.
Do not include any explanations or additional text within the %%% markers - just the JSON object."""
    
    prompt = f"{instruction}\n\n{emotion_mapping}\n\n{examples_text}\n\n{batch_data}\n\nPlease provide the results in the exact format described above and use %%% as markers."
    
    return prompt

# Rich background prompt with comprehensive information
def build_rich_background_prompt(batch_texts, examples, emotion_dict):
    """Optimized rich background prompt using patterns: Persona + ContextManager + Template + Recipe"""

    # Step 1: Role and Task
    role_description = """ROLE:
You are a professional emotion classification assistant trained on the GoEmotions dataset by Google.
Your task is to classify each text based on the **primary** emotion expressed in it.
You must reason carefully and only select one dominant emotion label per text."""

    # Step 2: Context Background
    dataset_info = """CONTEXT:
The GoEmotions dataset contains 58k Reddit comments annotated into 27 emotion classes + 1 neutral class.
It captures subtle, nuanced human emotions in natural language.
Use this context to guide your analysis and avoid surface-level cues only."""

    # Step 3: Emotion Mapping (Compact)
    emotion_mapping = "EMOTION MAPPING (name -> id):\n" + ", ".join(
        f"{emotion}:{idx}" for emotion, idx in emotion_dict.items()
    )

    # Step 4: Classification Guidelines (Recipe Pattern)
    guidelines = """CLASSIFICATION STEPS:
1. Read the full text and identify its dominant emotion.
2. Refer to the mapping and choose the appropriate emotion number.
3. If no strong emotion is expressed, use 'neutral'.
4. Ignore slang, sarcasm, emoji unless contextually necessary.
5. Do NOT guess — if uncertain, choose the best fitting label only."""

    # Step 5: Examples (Limited for clarity)
    examples_text = "ANNOTATED EXAMPLES:\n"
    for ex in examples[:8]:  # limit to 8 for conciseness
        emotion_name = next((name for name, idx in emotion_dict.items() if idx == ex["label"]), "unknown")
        examples_text += f"Text: {ex['text']}\nLabel: {ex['label']} ({emotion_name})\n---\n"

    # Step 6: Inputs to classify
    batch_data = "TEXTS TO CLASSIFY:\n"
    for i, text in enumerate(batch_texts):
        batch_data += f"{i+1}. {text}\n"

    # Step 7: Output Format Template
    output_format = """OUTPUT INSTRUCTIONS:
Return only a JSON object where each input text is a key and its emotion number is the value.
Example:
%%%
{"text1": 12, "text2": 7}
%%%

- Do not explain the classifications.
- Do not include any other text inside the %%% block.
- Make sure the number of JSON entries equals the number of input texts."""

    # Optional Reflection Pattern (for development/debug)
    reflection = """CHECKLIST (before responding):
- [ ] Did you include all input texts in the output JSON?
- [ ] Are all emotion values valid numbers from the mapping?
- [ ] Did you avoid extra explanation inside the %%% block?"""

    # Assemble full prompt
    prompt = (
        f"{role_description}\n\n"
        f"{dataset_info}\n\n"
        f"{emotion_mapping}\n\n"
        f"{guidelines}\n\n"
        f"{examples_text}\n\n"
        f"{batch_data}\n\n"
        f"{output_format}\n\n"
        f"{reflection}"
    )

    return prompt

# Dictionary of available prompt templates for easy access
prompt_templates = {
    "minimal": build_minimal_prompt,
    "basic": build_basic_prompt,
    "rich_background": build_rich_background_prompt
} 