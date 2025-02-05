# Step 1: Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 2: Load Dataset with Correct Encoding
# Replace 'Dataset.csv' with the path to your dataset
# Specify the correct encoding (e.g., 'latin1', 'ISO-8859-1', 'windows-1252')
df = pd.read_csv('Dataset.csv', encoding='latin1')

# Step 3: Handle Missing Values
# Check for missing values in the 'Complaint Text' column
print("Missing values in 'Complaint Text':", df['Complaint Text'].isnull().sum())

# Option A: Remove rows with missing values
df = df.dropna(subset=['Complaint Text'])

# Option B: Fill missing values with an empty string
# df['Complaint Text'] = df['Complaint Text'].fillna('')

# Step 4: Label Encoding
label_encoder = LabelEncoder()
df['department_encoded'] = label_encoder.fit_transform(df['Department'])

# Step 5: Feature Extraction (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)  # Limit features to 5000 for simplicity
X = tfidf.fit_transform(df['Complaint Text']).toarray()  # Use raw complaint text
y = df['department_encoded']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training (Logistic Regression)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 9: Save the Model and Vectorizer
joblib.dump(model, 'department_classifier.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Step 10: Predict Department for New Complaints
def predict_department(complaint_text):
    # Convert text to features
    text_features = tfidf.transform([complaint_text]).toarray()
    # Predict the department
    predicted_label = model.predict(text_features)
    # Decode the label
    predicted_department = label_encoder.inverse_transform(predicted_label)
    return predicted_department[0]

# Step 11: Take New Input from the User
while True:
    # Ask the user to enter a complaint
    user_complaint = input("Enter your complaint (or type 'exit' to quit): ")
    
    # Exit the loop if the user types 'exit'
    if user_complaint.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        break
    
    # Predict the department for the user's complaint
    try:
        department = predict_department(user_complaint)
        print(f"Predicted Department: {department}")
    except Exception as e:
        print(f"Error: {e}. Please try again.")