import pandas as pd
from googletrans import Translator

# Load the Excel file
df = pd.read_excel('test_caption_image.xlsx')

# Initialize translator
translator = Translator()

# Function to translate a single caption
def translate_caption(text):
    try:
        result = translator.translate(text, src='vi', dest='en')
        return result.text
    except Exception as e:
        print(f"Error translating '{text}': {e}")
        return text

# Apply translation to the 'caption' column
df['caption_en'] = df['caption'].apply(translate_caption)

# Save to a new Excel file
df.to_excel('test_caption_image_translated.xlsx', index=False)

print("Translation complete. Saved to 'test_caption_image_translated.xlsx'.")