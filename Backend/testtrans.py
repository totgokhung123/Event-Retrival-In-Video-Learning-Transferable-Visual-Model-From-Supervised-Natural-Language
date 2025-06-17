from deep_translator import GoogleTranslator

translated = GoogleTranslator(source='vi', target='en').translate("Tôi rất thích xem các bức ảnh đẹp về thiên nhiên , hi.")
print(translated)