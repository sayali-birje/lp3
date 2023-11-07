import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix

# Set seaborn style
sns.set()

# Load the dataset
dataset = pd.read_csv("Churn_Modelling.csv")

# Define the columns for features and target
X_columns = dataset.columns[3:13]  # Exclude CustomerId and Surname
Y_columns = dataset.columns[-1]

# Extract features and target
X = dataset[X_columns]
Y = dataset[Y_columns]

# Apply Label Encoding to 'Geography' and 'Gender'
label_encoder = LabelEncoder()
X['Geography'] = label_encoder.fit_transform(X['Geography'])
X['Gender'] = label_encoder.fit_transform(X['Gender'])

# Create a list of categorical column indices
categorical_columns = [1, 2]  # Assuming columns 1 (Geography) and 2 (Gender) are categorical

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(drop='first'), categorical_columns),
        ('scaler', StandardScaler(), list(set(range(10)) - set(categorical_columns)))
    ],
    remainder='passthrough'
)

# Create a pipeline for data preprocessing
pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

# Preprocess the features
X = pipeline.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Build the neural network model
classifier = Sequential()
classifier.add(Dense(6, activation='relu', input_dim=X_train.shape[1]))
classifier.add(Dropout(0.1))
classifier.add(Dense(6, activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(1, activation='sigmoid'))

# Compile the model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = classifier.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.1, verbose=2)

# Make predictions
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Calculate and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate and print the accuracy
accuracy = (cm[0, 0] + cm[1, 1]) / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

