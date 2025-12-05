# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.

## Program:
```
Program to implement the SVM For Spam Mail Detection.
Developed by: INESH N
RegisterNumber: 212223220036
```
```
# Step 1: Import Libraries
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

# Step 2: Detect Encoding
with open("spam.csv", 'rb') as f:
    result = chardet.detect(f.read())
encoding_used = result['encoding']
print("Detected encoding:", encoding_used)

# Step 3: Load Data
data = pd.read_csv("spam.csv", encoding=encoding_used)
print("Dataset loaded successfully.")

# Step 4: Inspect Data
print("\nData Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# Rename columns if necessary (common for spam datasets)
data.columns = ["label", "text"] + list(data.columns[2:])

# Step 5: Split Into x and y
x = data["text"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Step 6: Vectorization
cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

# Step 7: Train SVM Model
model = SVC()
model.fit(x_train_cv, y_train)

# Step 8: Predict
y_pred = model.predict(x_test_cv)

# Step 9: Model Evaluation
accuracy = metrics.accuracy_score(y_test, y_pred)
print("\nSVM Model Accuracy:", accuracy)
```


## Output:
<img width="460" height="600" alt="image" src="https://github.com/user-attachments/assets/be69d488-72db-4af1-8708-ef506112a2a7" />





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
