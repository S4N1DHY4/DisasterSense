import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")

def gen_email(highlighted_categories):
    model = genai.GenerativeModel("gemini-1.5-flash")
    categories_text = ", ".join(highlighted_categories[:-1])
    location = highlighted_categories[-1]
    chat = model.start_chat(history=[])
    prompt1 = f"Generate an email template with subject and body asking for help from disaster management organizations regarding the following {categories_text}."
    response1 = chat.send_message(prompt1)
    if location != 'No locations found':
        prompt2 = f"Also include the actions that should be taken by the Government or any other disaster management authority after the disaster has occured, based on the disaster management guidelines issued in that area {location}."
        response2 = chat.send_message(prompt2)
    return chat.history[-1].parts[0].text
