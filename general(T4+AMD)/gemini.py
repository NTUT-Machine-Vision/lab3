from google import genai


def generate_text(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """Generate text using Google Gemini AI."""
    try:
        client = genai.Client()
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Error generating text with model '{model}': {str(e)}") from e
