import ktrain

predictor = ktrain.load_predictor('/tmp/arabic_predictor')

print(predictor.predict('انا مهتم برياضة النوم'))