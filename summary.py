import google.generativeai as genai

def generate_summary(data, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    input_data = f"Summarise {data}"
    
    # Call the model
    response = model.generate_content(input_data)
    
    # Extract relevant content
    if hasattr(response, 'content'):
        summary_text = response.content  # Assuming the summary is in the 'content' field
    else:
        summary_text = "No content generated"

    return summary_text
