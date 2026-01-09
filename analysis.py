import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the dataset
try:
    df = pd.read_csv('online_shoppers_intention.csv')
    
    # Display first few rows
    print("First 5 rows:")
    print(df.head())
    print("\n" + "="*50 + "\n")
    
    # Display dataset shape
    print(f"Dataset Shape: {df.shape}")
    print("\n" + "="*50 + "\n")
    
    # Display column details
    print("Column Details:")
    print(df.info())
    print("\n" + "="*50 + "\n")
    
    # --- Data Preprocessing ---
    print("Starting Data Preprocessing...\n")

    # 1. Handle Missing Values
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Total missing values before cleaning: {missing_values}")
    
    # Drop duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Duplicates dropped: {initial_rows - df.shape[0]}")
    
    # Drop rows with missing values (if any remain)
    df.dropna(inplace=True)
    print(f"Shape after cleaning: {df.shape}")
    print("\n" + "-"*30 + "\n")

    # --- Exploratory Data Analysis (EDA) ---
    print("Starting Exploratory Data Analysis...")
    
    # 1. Plot Distributions of Key Features
    # We use some of the numerical columns we identified earlier
    eda_cols = ['Administrative_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    
    for col in eda_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'eda_dist_{col}.png')
        plt.close()
    print("Distribution plots saved.")

    # 2. Visualize Correlations
    # We'll use the numerical columns for correlation
    # Using the same eda_cols plus 'SpecialDay' and 'OperatingSystems' as examples, or just select all numerical
    corr_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # We should exclude the target if we haven't encoded it yet, but here it is boolean or object.
    # Revenue is boolean in original, but let's check. 
    # 'Revenue' is still boolean/object here before step 2. 
    # So we'll select types carefully or just use the specific list + others.
    
    numeric_df = df.select_dtypes(include=['float64', 'int64', 'int32'])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('eda_heatmap.png')
        plt.close()
        print("Correlation heatmap saved.")
    else:
        print("No numerical columns found for correlation heatmap.")

    # 3. Identify Customer Behavior Patterns
    # Revenue vs PageValues
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Revenue', y='PageValues', data=df)
    plt.title('PageValues vs. Revenue')
    plt.savefig('eda_pattern_PageValues_vs_Revenue.png')
    plt.close()

    # Revenue vs ExitRates
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Revenue', y='ExitRates', data=df)
    plt.title('ExitRates vs. Revenue')
    plt.savefig('eda_pattern_ExitRates_vs_Revenue.png')
    plt.close()
    
    print("Pattern plots saved.")
    
    # 4. Textual Insights for Pattern Identification
    print("\n--- Insights: average values by Revenue (0=No Purchase, 1=Purchase) ---")
    print(df.groupby('Revenue')[['PageValues', 'ExitRates', 'BounceRates', 'ProductRelated']].mean())
    print("-" * 30 + "\n")
    
    print("\n" + "-"*30 + "\n")

    # 2. Encode Categorical Variables
    # Convert 'Weekend' and 'Revenue' to integers (0/1)
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    
    # One-Hot Encoding for 'Month' and 'VisitorType'
    categorical_cols = ['Month', 'VisitorType']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("Data after Encoding:")
    print(df.head())
    print("\n" + "-"*30 + "\n")

    # 3. Scale Numerical Features
    # Identify numerical columns to scale
    # We exclude the target 'Revenue', 'Weekend' (is binary now), and the newly created dummy columns
    # Actually, simpler approach: Scale specific known numerical columns
    numerical_cols = [
        'Administrative', 'Administrative_Duration', 
        'Informational', 'Informational_Duration', 
        'ProductRelated', 'ProductRelated_Duration', 
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
    ]
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    print("First 5 rows after Scaling:")
    print(df[numerical_cols].head())
    print("\n" + "-"*30 + "\n")

    # 4. Split Data into Training and Testing Sets
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Data Splitting Complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("\n" + "="*50 + "\n")

    # --- Model Building and Evaluation ---
    print("Starting Model Training and Evaluation...")

    def evaluate_model(model, X_test, y_test, model_name):
        print(f"--- Evaluating {model_name} ---")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'cm_{model_name.replace(" ", "_")}.png')
        plt.close()
        print(f"Confusion Matrix saved to 'cm_{model_name.replace(' ', '_')}.png'")
        print("-" * 30 + "\n")
        
        return {'Model': model_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

    # 1. Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # 2. Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # --- Model Comparison ---
    print("--- Final Model Comparison ---")
    results_df = pd.DataFrame([lr_metrics, rf_metrics])
    print(results_df)
    
    # Select Best Model based on F1-Score
    best_model_row = results_df.loc[results_df['F1'].idxmax()]
    print(f"\nBest Performing Model: {best_model_row['Model']} with F1-Score: {best_model_row['F1']:.4f}")
    
    print("\n" + "="*50 + "\n")
    print("Error: File 'online_shoppers_intention.csv' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
