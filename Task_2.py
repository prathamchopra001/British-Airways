import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve, auc
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set the aesthetic style of the plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
colors = sns.color_palette("deep")

print("# 1. Data Loading and Initial Exploration")
print("-----------------------------------------")

# Load the data
df = pd.read_csv('data/customer_booking.csv', encoding='latin-1')

# Display basic information
print(f"Dataset shape: {df.shape} (rows, columns)")
print("\nColumn information:")
for col in df.columns:
    print(f"- {col}: {df[col].dtype}")

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")

# Display basic statistics
print("\nBasic statistics for numerical columns:")
print(df.describe())

# Target variable distribution
print("\nTarget variable (booking_complete) distribution:")
target_counts = df['booking_complete'].value_counts()
target_percentage = df['booking_complete'].value_counts(normalize=True) * 100
for value, count in target_counts.items():
    print(f"- {value}: {count} ({target_percentage[value]:.2f}%)")

print("\n# 2. Exploratory Data Analysis")
print("-----------------------------------------")

# Analyze categorical variables
categorical_cols = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']
print("\nUnique values for categorical columns:")
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"- {col}: {unique_count} unique values")
    if unique_count < 10:  # Only show all values if there are fewer than 10
        print(f"  Values: {sorted(df[col].unique())}")
    else:
        print(f"  Top 5 frequent values: {df[col].value_counts().nlargest(5).index.tolist()}")

# Analyze numerical variables
numerical_cols = ['num_passengers', 'purchase_lead', 'length_of_stay', 'flight_hour', 
                 'flight_duration', 'wants_extra_baggage', 'wants_preferred_seat', 
                 'wants_in_flight_meals']

# Correlation analysis for numerical variables
corr_matrix = df[numerical_cols + ['booking_complete']].corr()
print("\nCorrelation with target variable (booking_complete):")
print(corr_matrix['booking_complete'].sort_values(ascending=False))

# Create visualization functions
def create_eda_visualizations(df):
    """Create exploratory data analysis visualizations"""
    
    # Visualize target variable distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='booking_complete', data=df)
    plt.title('Distribution of Booking Completion')
    plt.xlabel('Booking Complete (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    
    # Add count and percentage annotations
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                f'{height} ({height/total*100:.1f}%)',
                ha="center")
    
    plt.tight_layout()
    plt.savefig('visuals/Task-2/booking_completion_distribution.png')
    plt.close()
    
    # Visualize correlations
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=.5)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig('visuals/Task-2/correlation_matrix.png')
    plt.close()
    
    # Analyze categorical variables with respect to target
    plt.figure(figsize=(15, 20))
    plot_idx = 1
    
    for col in categorical_cols:
        if df[col].nunique() <= 10:  # Only plot if there aren't too many categories
            plt.subplot(3, 2, plot_idx)
            order = df.groupby(col)['booking_complete'].mean().sort_values(ascending=False).index
            sns.barplot(x=col, y='booking_complete', data=df, order=order)
            plt.title(f'Booking Completion Rate by {col}')
            plt.ylabel('Completion Rate')
            plt.xticks(rotation=45)
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('visuals/Task-2/categorical_vs_target.png')
    plt.close()
    
    # Booking completion by number of passengers
    plt.figure(figsize=(10, 6))
    completion_by_passengers = df.groupby('num_passengers')['booking_complete'].mean().reset_index()
    ax = sns.barplot(x='num_passengers', y='booking_complete', data=completion_by_passengers)
    plt.title('Booking Completion Rate by Number of Passengers')
    plt.xlabel('Number of Passengers')
    plt.ylabel('Completion Rate')
    
    # Add count annotations
    counts = df.groupby('num_passengers').size().values
    for i, p in enumerate(ax.patches):
        ax.annotate(f'n={counts[i]}', (p.get_x() + p.get_width()/2., p.get_height() + 0.01),
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('visuals/Task-2/completion_by_passengers.png')
    plt.close()
    
    # Analyze booking additional services
    services = ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']
    plt.figure(figsize=(12, 5))
    
    for i, service in enumerate(services, 1):
        plt.subplot(1, 3, i)
        sns.barplot(x=service, y='booking_complete', data=df)
        plt.title(f'Completion Rate by {service}')
        plt.xlabel('Requested (0 = No, 1 = Yes)')
        plt.ylabel('Completion Rate')
    
    plt.tight_layout()
    plt.savefig('visuals/Task-2/completion_by_services.png')
    plt.close()
    
    # Analyze purchase lead time vs completion
    plt.figure(figsize=(10, 6))
    # Create bins for purchase lead
    df['lead_bins'] = pd.cut(df['purchase_lead'], 
                            bins=[0, 7, 30, 90, df['purchase_lead'].max()],
                            labels=['0-7 days', '8-30 days', '31-90 days', '90+ days'])
    
    sns.barplot(x='lead_bins', y='booking_complete', data=df)
    plt.title('Booking Completion Rate by Purchase Lead Time')
    plt.xlabel('Days Before Departure')
    plt.ylabel('Completion Rate')
    plt.tight_layout()
    plt.savefig('visuals/Task-2/completion_by_lead_time.png')
    plt.close()
    
    # Analyze flight duration vs completion
    plt.figure(figsize=(10, 6))
    # Create bins for flight duration - fixed to be monotonically increasing
    df['duration_bins'] = pd.cut(df['flight_duration'], 
                                bins=[-float('inf'), 2, 5, 10, float('inf')],  # Changed this line
                                labels=['0-2 hours', '2-5 hours', '5-10 hours', '10+ hours'])
    
    sns.barplot(x='duration_bins', y='booking_complete', data=df)
    plt.title('Booking Completion Rate by Flight Duration')
    plt.xlabel('Flight Duration')
    plt.ylabel('Completion Rate')
    plt.tight_layout()
    plt.savefig('visuals/Task-2/completion_by_duration.png')
    plt.close()
    
    return

# Generate EDA visualizations
create_eda_visualizations(df)

print("\n# 3. Feature Engineering")
print("-----------------------------------------")

def engineer_features(df):
    """Create new features to enhance model performance"""
    
    # Create a copy of the DataFrame
    data = df.copy()
    
    # 1. Categorize purchase lead time
    data['lead_time_category'] = pd.cut(
        data['purchase_lead'], 
        bins=[0, 7, 30, 90, float('inf')],
        labels=['Last minute', 'Short notice', 'Regular', 'Early bird']
    )
    
    # 2. Extract route distance (number of segments)
    data['route_segments'] = data['route'].apply(lambda x: len(x.split(' -> ')))
    
    # 3. Create a weekend flight indicator
    data['is_weekend'] = data['flight_day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    
    # 4. Create time of day categories
    data['time_of_day'] = pd.cut(
        data['flight_hour'],
        bins=[-1, 5, 11, 16, 21, 24],
        labels=['Late Night/Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
    )
    
    
    # 5. Create a feature for peak flight hours (morning 6-9, evening 17-20)
    data['is_peak_hour'] = data['flight_hour'].apply(
        lambda x: 1 if (6 <= x <= 9) or (17 <= x <= 20) else 0
    )
    
    # 6. Create a feature for booking "complexity" - sum of additional services
    data['booking_complexity'] = data['wants_extra_baggage'] + \
                               data['wants_preferred_seat'] + \
                               data['wants_in_flight_meals']
    
    # 7. Categorize flight duration
    data['flight_duration_category'] = pd.cut(
        data['flight_duration'],
        bins=[0, 2, 5, 10, float('inf')],
        labels=['Short', 'Medium', 'Long', 'Ultra-long']
    )
    
    # 8. Group similar routes by distance (approximation)
    data['route_distance'] = pd.cut(
        data['flight_duration'],
        bins=[0, 3, 6, 10, float('inf')],
        labels=['Short haul', 'Medium haul', 'Long haul', 'Ultra long haul']
    )
    
    # 9. Create a feature for international vs. domestic (simplified)
    # This is a simplification assuming routes with certain patterns are international
    data['is_international'] = data['route'].apply(
        lambda x: 1 if ' -> ' in x and x.split(' -> ')[0] != x.split(' -> ')[-1][:2] else 0
    )
    
    # 10. Booking from origin country (matching origin of flight)
    data['booking_from_origin'] = (
        data['booking_origin'] == data['route'].apply(lambda x: x.split(' -> ')[0])
    ).astype(int)
    
    # Print the new features
    print("New features created:")
    for feature in ['lead_time_category', 'route_segments', 'is_weekend', 'time_of_day', 
                   'is_peak_hour', 'booking_complexity', 'flight_duration_category',
                   'route_distance', 'is_international', 'booking_from_origin']:
        print(f"- {feature}")
    
    return data

# Apply feature engineering
df_engineered = engineer_features(df)

# Display the engineered features
print("\nSample of engineered dataset:")
print(df_engineered.sample(5).T)

print("\n# 4. Data Preprocessing")
print("-----------------------------------------")

def preprocess_data(df):
    """Preprocess data for model training"""
    
    # Identify categorical and numerical columns
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' or 
                       df[col].dtype.name == 'category']
    numerical_cols = [col for col in df.columns if col not in categorical_cols and 
                     col != 'booking_complete']
    
    # Print columns
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    
    # Split the dataset into features and target
    X = df.drop('booking_complete', axis=1)
    y = df['booking_complete']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create preprocessing pipelines
    # For numerical features: imputation + scaling
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # For categorical features: imputation + one-hot encoding
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ]
    )
    
    # Fit the preprocessor on the training data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    print(f"\nPreprocessed training data shape: {X_train_preprocessed.shape}")
    print(f"Preprocessed test data shape: {X_test_preprocessed.shape}")
    
    # Get feature names after preprocessing
    ohe_feature_names = []
    if categorical_cols:
        ohe_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    
    feature_names = list(numerical_cols) + list(ohe_feature_names)
    
    return preprocessor, X_train, X_test, X_train_preprocessed, X_test_preprocessed, y_train, y_test, feature_names

# Preprocess the data
preprocessor, X_train, X_test, X_train_preprocessed, X_test_preprocessed, y_train, y_test, feature_names = preprocess_data(df_engineered)

print("\n# 5. Model Training and Evaluation")
print("-----------------------------------------")

def train_and_evaluate_model(X_train, y_train, X_test, y_test, feature_names):
    """Train a Random Forest model and evaluate its performance"""
    
    # Initialize the model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest set accuracy: {accuracy:.4f}")
    print(f"Test set ROC AUC: {roc_auc:.4f}")
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    print(f"Cross-validation ROC AUC scores: {cv_scores}")
    print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('visuals/Task-2/confusion_matrix.png')
    plt.close()
    
    # Create ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('visuals/Task-2/roc_curve.png')
    plt.close()
    
    # Feature importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    top_n = 15
    plt.title(f'Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[indices[:top_n]], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visuals/Task-2/feature_importance.png')
    plt.close()
    
    # Print top 10 features
    print("\nTop 10 most important features:")
    for i in range(10):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return rf_model, importances, indices

# Train and evaluate the model
model, importances, importance_indices = train_and_evaluate_model(
    X_train_preprocessed, y_train, X_test_preprocessed, y_test, feature_names
)

print("\n# 6. Summary of Findings")
print("-----------------------------------------")

# Create a summary of the findings
print("Key findings from the analysis:")
print("1. Overall booking completion rate:", f"{df['booking_complete'].mean()*100:.1f}%")
print("2. Most important predictors of booking completion:")
for i in range(5):
    print(f"   - {feature_names[importance_indices[i]]}")
print("3. Model performance:")
print(f"   - Accuracy: {accuracy_score(y_test, model.predict(X_test_preprocessed)):.4f}")
print(f"   - ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test_preprocessed)[:, 1]):.4f}")

print("\nSuggested actions to improve booking completion rates:")
print("1. Focus on optimizing the booking process for the most influential factors")
print("2. Consider targeted interventions for customer segments with lower completion rates")
print("3. Use the predictive model to identify bookings at risk of abandonment")
print("4. A/B test different approaches to address the key factors influencing booking completion")