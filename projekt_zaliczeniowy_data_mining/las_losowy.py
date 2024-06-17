import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report




file_path = 'C:/Users/j.trzcinski/Documents/python_julien/dane_filtered_final.csv'
df = pd.read_csv(file_path, encoding='utf-8')

print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Loaded data")


columns_to_drop = ['user_id', 'first_name', 'last_name', 'email', 'is_returned_or_cancelled']
X = df.drop(columns=columns_to_drop)
y = df['is_returned_or_cancelled']


numerical_features = ['age', 'avg_num_of_item', 'avg_total_sales']

categorical_features = ['gender', 'state', 'street_address', 'city', 'country', 'traffic_source', 'postal_code']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

smote = SMOTE(random_state=37)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)




pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', RandomForestClassifier(random_state=37))
])

print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - grid search start")


param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [None],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__max_features': ['sqrt']
}



grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

refined_param_grid = {
    'classifier__n_estimators': [best_params['classifier__n_estimators']],
    'classifier__max_depth': [best_params['classifier__max_depth']],
    'classifier__min_samples_split': [2, 5, 10],  # You can adjust these values if needed
    'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf']],
    'classifier__max_features': [best_params['classifier__max_features']]
}

grid_search = GridSearchCV(pipeline, refined_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-validation Accuracy: {:.2f}".format(grid_search.best_score_))

# Evaluate best model on test set
test_accuracy = grid_search.best_estimator_.score(X_test, y_test)
print("Test Set Accuracy with Best Parameters: {:.2f}".format(test_accuracy))


# Predictions on test set
y_pred = grid_search.best_estimator_.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC score
y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score: {:.2f}".format(roc_auc))

pr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()