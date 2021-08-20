# pip install ktrain
import ktrain
import pandas as pd
from ktrain import text 
#################
df = pd.read_csv('SANAD.csv')

# Check if the dataset is balanced or not
print(df['Topic Category'].value_counts())

# Split the dataset into train and test
df_test = df.sample(frac=0.15, random_state=42)
df_train = df.drop(df_test.index)

# Convert from Series to list
x_train = df_train['Post Body'].tolist()
y_train = df_train['Topic Category'].to_numpy()
x_test = df_test['Post Body'].tolist()
y_test = df_test['Topic Category'].to_numpy()

# You can select best option for you data from here: https://github.com/aub-mind/arabert
MODEL_NAME = 'aubmindlab/bert-base-arabertv02'

# Specify the class names to train as multi label classification
class_names = df['Topic Category'].unique()

# Instantiate a transfomer and giving it the model name, class names and max length
transfomer = text.Transformer(MODEL_NAME, class_names= class_names, maxlen=90)

# Preprocess the data
train = transfomer.preprocess_train(x_train, y_train)
validation = transfomer.preprocess_test(x_test, y_test)

# Get a classifier from the transfomer
model = transfomer.get_classifier()

# Get learner to train (finetune) the pretrained model
learner = ktrain.get_learner(model, train_data=train, val_data=validation)

# It may need long time but ensures healthy training
learner.lr_find(show_plot=True, max_epochs=2)

# After the learning rate find process plot the curve of leaning rate - loss to select best learning rate
learner.lr_plot()

# Fit the leraner to the data for 2 epochs
# Many other fit options available on the documentation of ktrain https://github.com/amaiya/ktrain
learner.autofit(1e-5, 2)
learner.validate(class_names=transfomer.get_classes())

predictor = ktrain.get_predictor(learner.model, transfomer)
predictor.save('/content/drive/MyDrive/NLP/my_predictor')