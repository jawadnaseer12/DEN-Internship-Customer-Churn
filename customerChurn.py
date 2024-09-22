try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv('customer_churn_dataset-training-master.csv')
    print('\nData Before Cleaning\n')
    print(df.head())
    df.dropna(inplace=True)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Subscription Type'] = le.fit_transform(df['Subscription Type'])
    df['Contract Length'] = le.fit_transform(df['Contract Length'])

    print('\nData After Cleaning\n')
    print(df.head())

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    print(f'\nAccuracy: {accuracy_score(y_test, y_predict)}')
    # print(classification_report(y_test, y_predict))
    print(f'AUC-ROC Score: {roc_auc_score(y_test, y_predict)}')

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    features = X.columns

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[indices], y=features[indices], hue=features[indices], palette="Blues_d", legend=False)
    plt.title("Feature Importance")
    plt.show()

except ImportError as e:
    print(f"Debugging failed: {e}")

except Exception as e:
    print(f"Error: {e}")