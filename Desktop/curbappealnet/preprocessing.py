# ============================================================================
# SATELLITE IMAGERY BASED PROPERTY VALUATION - PREPROCESSING
# ============================================================================

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")

# ============================================================================
# 2. LOAD DATA
# ============================================================================

train_df = pd.read_csv('train_curbappealnet.csv')
test_df = pd.read_csv('test_curbappealnet.csv')
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"\nNote: Test data has no 'price' column (to be predicted)")
print("\n" + "="*80)
print("DATA INFORMATION")
print("="*80)
print(train_df.info())
print("\nMissing values:")
print(train_df.isnull().sum())

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Statistical summary
print("\nStatistical Summary:")
print(train_df.describe())

# --- Price Distribution ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(train_df['price'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.hist(np.log1p(train_df['price']), bins=50, edgecolor='black', alpha=0.7, color='green')
plt.xlabel('Log(Price + 1)')
plt.ylabel('Frequency')
plt.title('Log-Transformed Price Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.boxplot(train_df['price'])
plt.ylabel('Price')
plt.title('Price Boxplot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPrice Statistics:")
print(f"Mean: ${train_df['price'].mean():,.2f}")
print(f"Median: ${train_df['price'].median():,.2f}")
print(f"Std Dev: ${train_df['price'].std():,.2f}")
print(f"Min: ${train_df['price'].min():,.2f}")
print(f"Max: ${train_df['price'].max():,.2f}")

# --- Correlation Analysis ---
plt.figure(figsize=(14, 10))
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
correlation_matrix = train_df[numeric_cols].corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Top correlations with price
price_corr = correlation_matrix['price'].sort_values(ascending=False)
print("\nTop 10 features correlated with price:")
print(price_corr[1:11])

# --- Feature Distributions ---
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
features_to_plot = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                    'floors', 'view', 'condition', 'grade', 'yr_built']

for idx, feature in enumerate(features_to_plot):
    row, col = idx // 3, idx % 3
    axes[row, col].hist(train_df[feature], bins=30, edgecolor='black', alpha=0.7)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].set_title(f'{feature} Distribution')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Geographic Analysis ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(train_df['long'], train_df['lat'], 
                     c=train_df['price'], cmap='viridis', 
                     alpha=0.5, s=10)
plt.colorbar(scatter, label='Price')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Property Locations Colored by Price')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
zipcode_price = train_df.groupby('zipcode')['price'].mean().sort_values(ascending=False).head(15)
plt.barh(range(len(zipcode_price)), zipcode_price.values)
plt.yticks(range(len(zipcode_price)), zipcode_price.index)
plt.xlabel('Average Price')
plt.ylabel('Zipcode')
plt.title('Top 15 Zipcodes by Average Price')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('geographic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Price vs Key Features ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Price vs Sqft Living
axes[0, 0].scatter(train_df['sqft_living'], train_df['price'], alpha=0.5, s=10)
axes[0, 0].set_xlabel('Sqft Living')
axes[0, 0].set_ylabel('Price')
axes[0, 0].set_title('Price vs Sqft Living')
axes[0, 0].grid(True, alpha=0.3)

# Price vs Grade
train_df.boxplot(column='price', by='grade', ax=axes[0, 1])
axes[0, 1].set_xlabel('Grade')
axes[0, 1].set_ylabel('Price')
axes[0, 1].set_title('Price vs Grade')
axes[0, 1].get_figure().suptitle('')

# Price vs Waterfront
train_df.boxplot(column='price', by='waterfront', ax=axes[1, 0])
axes[1, 0].set_xlabel('Waterfront')
axes[1, 0].set_ylabel('Price')
axes[1, 0].set_title('Price vs Waterfront')
axes[1, 0].get_figure().suptitle('')

# Price vs View
train_df.boxplot(column='price', by='view', ax=axes[1, 1])
axes[1, 1].set_xlabel('View')
axes[1, 1].set_ylabel('Price')
axes[1, 1].set_title('Price vs View')
axes[1, 1].get_figure().suptitle('')

plt.tight_layout()
plt.savefig('price_vs_features.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. LOAD AND EXPLORE SATELLITE IMAGES
# ============================================================================

print("\n" + "="*80)
print("SATELLITE IMAGES ANALYSIS")
print("="*80)

# Define image paths - UPDATED TEST PATH
TRAIN_IMAGE_DIR = '1767547761150_satellite_ima'  
TEST_IMAGE_DIR = r'C:\dataset_curbappealnet\satellite_ima\test_images'   # UPDATED PATH
IMG_SIZE = 224

def get_image_path(property_id, image_dir):
    """Get image path for a given property ID"""
    for ext in ['.jpg', '.png', '.jpeg']:
        img_path = os.path.join(image_dir, f"{int(property_id)}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None

# Check available TRAINING images
available_images = []
missing_images = []

print("Checking TRAINING image availability...")
for idx, property_id in enumerate(train_df['id']):
    img_path = get_image_path(property_id, TRAIN_IMAGE_DIR)
    if img_path:
        available_images.append((property_id, img_path))
    else:
        missing_images.append(property_id)
    
    if (idx + 1) % 1000 == 0:
        print(f"Processed {idx + 1}/{len(train_df)} properties...")

print(f"\nTotal properties: {len(train_df)}")
print(f"Available images: {len(available_images)}")
print(f"Missing images: {len(missing_images)}")
print(f"Coverage: {len(available_images)/len(train_df)*100:.2f}%")

train_df_with_images = train_df[train_df['id'].isin([img[0] for img in available_images])].copy()
print(f"Filtered training dataset shape: {train_df_with_images.shape}")

# Check TEST images availability
print("\n" + "="*80)
print("Checking TEST images availability...")
test_available_images = []
test_missing_images = []

for idx, property_id in enumerate(test_df['id']):
    img_path = get_image_path(property_id, TEST_IMAGE_DIR)
    if img_path:
        test_available_images.append((property_id, img_path))
    else:
        test_missing_images.append(property_id)
    
    if (idx + 1) % 1000 == 0:
        print(f"Processed {idx + 1}/{len(test_df)} test properties...")

print(f"\nTest Dataset:")
print(f"Total test properties: {len(test_df)}")
print(f"Available test images: {len(test_available_images)}")
print(f"Missing test images: {len(test_missing_images)}")
print(f"Coverage: {len(test_available_images)/len(test_df)*100:.2f}%")

test_df_with_images = test_df[test_df['id'].isin([img[0] for img in test_available_images])].copy()
print(f"Filtered test dataset shape: {test_df_with_images.shape}")

# Display sample images
print("\nDisplaying sample satellite images...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
sample_indices = np.random.choice(len(available_images), 12, replace=False)

for idx, ax in enumerate(axes.flat):
    property_id, img_path = available_images[sample_indices[idx]]
    img = Image.open(img_path)
    ax.imshow(img)
    
    price = train_df_with_images[train_df_with_images['id'] == property_id]['price'].values[0]
    ax.set_title(f'ID: {int(property_id)}\nPrice: ${price:,.0f}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_satellite_images.png', dpi=300, bbox_inches='tight')
plt.show()

# Check image dimensions
print("\nAnalyzing image dimensions...")
image_dimensions = []
for i, (property_id, img_path) in enumerate(available_images[:100]):
    img = Image.open(img_path)
    image_dimensions.append(img.size)

unique_dimensions = set(image_dimensions)
print(f"Unique image dimensions found: {unique_dimensions}")
print(f"Most common dimension: {max(set(image_dimensions), key=image_dimensions.count)}")

# ============================================================================
# 5. FEATURE ENGINEERING FROM TABULAR DATA
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING - TABULAR DATA")
print("="*80)

def engineer_features(df):
    """Create new features from existing ones"""
    df = df.copy()
    
    # Age of property
    df['age'] = 2025 - df['yr_built']
    df['age_after_renovation'] = np.where(df['yr_renovated'] > 0, 
                                          2025 - df['yr_renovated'], 
                                          df['age'])
    
    # Boolean features
    df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
    
    # Ratio features
    if 'price' in df.columns:
        df['price_per_sqft'] = df['price'] / df['sqft_living']
    df['sqft_living_ratio'] = df['sqft_living'] / df['sqft_lot']
    df['sqft_above_ratio'] = df['sqft_above'] / df['sqft_living']
    df['bedroom_bathroom_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
    
    # Living space features
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['sqft_per_room'] = df['sqft_living'] / (df['total_rooms'] + 1)
    
    # Neighborhood features
    df['neighborhood_quality'] = (df['sqft_living15'] + df['sqft_lot15']) / 2
    
    # Location-based features
    df['lat_long_interaction'] = df['lat'] * df['long']
    
    # Binning features
    if 'price' in df.columns:
        df['price_category'] = pd.cut(df['price'], 
                                       bins=[0, 200000, 400000, 600000, 1000000, float('inf')],
                                       labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    df['grade_category'] = pd.cut(df['grade'],
                                   bins=[0, 6, 8, 10, float('inf')],
                                   labels=['poor', 'average', 'good', 'excellent'])
    
    return df

# Apply feature engineering to training data
train_df_engineered = engineer_features(train_df_with_images)
print(f"Training - Original features: {train_df_with_images.shape[1]}")
print(f"Training - After feature engineering: {train_df_engineered.shape[1]}")
print(f"\nNew features created:")
new_features = set(train_df_engineered.columns) - set(train_df_with_images.columns)
for feat in new_features:
    print(f"  - {feat}")

# Apply same feature engineering to test data
print("\n" + "="*80)
print("FEATURE ENGINEERING - TEST DATA")
print("="*80)

# Add dummy price column for test data (needed for engineer_features function)
test_df_with_images['price'] = 0
test_df_engineered = engineer_features(test_df_with_images)
# Remove price-dependent features
test_df_engineered = test_df_engineered.drop(['price', 'price_per_sqft', 'price_category'], axis=1)

print(f"Test - After feature engineering: {test_df_engineered.shape[1]}")

# Check for infinite values
print("\nChecking for infinite values in training data...")
inf_cols = []
for col in train_df_engineered.select_dtypes(include=[np.number]).columns:
    if np.isinf(train_df_engineered[col]).any():
        inf_cols.append(col)
        print(f"  - {col}: {np.isinf(train_df_engineered[col]).sum()} infinite values")

# Replace infinite values with NaN and then fill
if inf_cols:
    train_df_engineered[inf_cols] = train_df_engineered[inf_cols].replace([np.inf, -np.inf], np.nan)
    train_df_engineered[inf_cols] = train_df_engineered[inf_cols].fillna(train_df_engineered[inf_cols].median())

# ============================================================================
# 6. IMAGE PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("IMAGE PREPROCESSING")
print("="*80)

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def create_image_dataset(df, image_dir, img_size=224):
    """Create image dataset from dataframe"""
    images = []
    valid_indices = []
    
    print(f"Loading {len(df)} images...")
    for idx, row in df.iterrows():
        img_path = get_image_path(row['id'], image_dir)
        if img_path:
            img_array = load_and_preprocess_image(img_path, target_size=(img_size, img_size))
            if img_array is not None:
                images.append(img_array)
                valid_indices.append(idx)
        
        if (len(images)) % 500 == 0:
            print(f"  Loaded {len(images)} images...")
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images), valid_indices

# Load training images
X_images, valid_indices = create_image_dataset(train_df_engineered, TRAIN_IMAGE_DIR, IMG_SIZE)
print(f"\nTraining image dataset shape: {X_images.shape}")

# Filter dataframe to match loaded images
train_df_final = train_df_engineered.loc[valid_indices].reset_index(drop=True)
print(f"Final training dataset shape: {train_df_final.shape}")

# Load test images
print("\n" + "="*80)
print("LOADING TEST IMAGES")
print("="*80)

X_images_test, valid_test_indices = create_image_dataset(test_df_engineered, TEST_IMAGE_DIR, IMG_SIZE)
print(f"\nTest image dataset shape: {X_images_test.shape}")

# Filter test dataframe
test_df_final = test_df_engineered.loc[valid_test_indices].reset_index(drop=True)
print(f"Final test dataset shape: {test_df_final.shape}")

# Visualize preprocessed images
print("\nVisualizing preprocessed images...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
sample_indices = np.random.choice(len(X_images), 8, replace=False)

for idx, ax in enumerate(axes.flat):
    ax.imshow(X_images[sample_indices[idx]])
    price = train_df_final.iloc[sample_indices[idx]]['price']
    ax.set_title(f'Price: ${price:,.0f}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('preprocessed_images.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. DEEP FEATURE EXTRACTION FROM IMAGES
# ============================================================================

print("\n" + "="*80)
print("DEEP FEATURE EXTRACTION")
print("="*80)

def extract_features_with_model(images, model_name='efficientnet'):
    """Extract deep features from images using pre-trained models"""
    
    print(f"\nExtracting features using {model_name}...")
    
    if model_name == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        preprocess_func = efficientnet_preprocess
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Preprocess images for the specific model
    processed_images = preprocess_func(images * 255.0)  # Convert back to [0, 255]
    
    # Extract features in batches
    batch_size = 32
    features_list = []
    
    for i in range(0, len(processed_images), batch_size):
        batch = processed_images[i:i+batch_size]
        batch_features = base_model.predict(batch, verbose=0)
        features_list.append(batch_features)
        
        if (i + batch_size) % 500 == 0:
            print(f"  Processed {min(i+batch_size, len(processed_images))}/{len(processed_images)} images...")
    
    features = np.vstack(features_list)
    print(f"Extracted features shape: {features.shape}")
    
    return features

# Extract features from TRAINING images
print("Using EfficientNetB0 for feature extraction on TRAINING images...")
image_features = extract_features_with_model(X_images, model_name='efficientnet')

# Create feature names
image_feature_names = [f'img_feat_{i}' for i in range(image_features.shape[1])]
image_features_df = pd.DataFrame(image_features, columns=image_feature_names)

print(f"\nTraining image features shape: {image_features_df.shape}")
print(f"Sample features:")
print(image_features_df.head())

# Extract features from TEST images
print("\n" + "="*80)
print("EXTRACTING FEATURES FROM TEST IMAGES")
print("="*80)

print("Using EfficientNetB0 for feature extraction on TEST images...")
test_image_features = extract_features_with_model(X_images_test, model_name='efficientnet')

# Create test image features dataframe
test_image_features_df = pd.DataFrame(test_image_features, columns=image_feature_names)

print(f"\nTest image features shape: {test_image_features_df.shape}")
print(f"Sample features:")
print(test_image_features_df.head())

# ============================================================================
# 8. COMBINE FEATURES AND PREPARE FINAL DATASET
# ============================================================================

print("\n" + "="*80)
print("COMBINING FEATURES - TRAINING DATA")
print("="*80)

# Select relevant tabular features
tabular_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                   'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                   'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
                   'lat', 'long', 'sqft_living15', 'sqft_lot15',
                   'age', 'age_after_renovation', 'is_renovated', 'has_basement',
                   'sqft_living_ratio', 'sqft_above_ratio', 'bedroom_bathroom_ratio',
                   'total_rooms', 'sqft_per_room', 'neighborhood_quality',
                   'lat_long_interaction']

# Combine tabular and image features for TRAINING
X_tabular = train_df_final[tabular_features].copy()
y = train_df_final['price'].values
property_ids = train_df_final['id'].values

# Handle missing values
print("\nHandling missing values in training tabular features...")
X_tabular = X_tabular.fillna(X_tabular.median())

# Standardize tabular features (FIT on training data)
scaler = StandardScaler()
X_tabular_scaled = scaler.fit_transform(X_tabular)
X_tabular_scaled_df = pd.DataFrame(X_tabular_scaled, columns=tabular_features)

# Combine all features for training
X_combined = pd.concat([X_tabular_scaled_df, image_features_df], axis=1)

print(f"\nTraining - Final feature set:")
print(f"  - Tabular features: {len(tabular_features)}")
print(f"  - Image features: {image_features_df.shape[1]}")
print(f"  - Total features: {X_combined.shape[1]}")
print(f"  - Samples: {X_combined.shape[0]}")

# Process TEST data
print("\n" + "="*80)
print("COMBINING FEATURES - TEST DATA")
print("="*80)

# Combine tabular and image features for TEST
X_tabular_test = test_df_final[tabular_features].copy()
test_property_ids = test_df_final['id'].values

# Handle missing values (use training median)
print("\nHandling missing values in test tabular features...")
for col in X_tabular_test.columns:
    if X_tabular_test[col].isnull().any():
        X_tabular_test[col] = X_tabular_test[col].fillna(X_tabular[col].median())

# Standardize test tabular features (TRANSFORM using training scaler - DO NOT FIT)
X_tabular_test_scaled = scaler.transform(X_tabular_test)
X_tabular_test_scaled_df = pd.DataFrame(X_tabular_test_scaled, columns=tabular_features)

# Combine all features for test
X_combined_test = pd.concat([X_tabular_test_scaled_df, test_image_features_df], axis=1)

print(f"\nTest - Final feature set:")
print(f"  - Tabular features: {len(tabular_features)}")
print(f"  - Image features: {test_image_features_df.shape[1]}")
print(f"  - Total features: {X_combined_test.shape[1]}")
print(f"  - Samples: {X_combined_test.shape[0]}")

# ============================================================================
# 9. TRAIN-VALIDATION SPLIT (FROM TRAINING DATA ONLY)
# ============================================================================

print("\n" + "="*80)
print("TRAIN-VALIDATION SPLIT (FROM TRAINING DATA ONLY)")
print("="*80)

# Split training data (80-20) for model validation during training
X_train, X_val, y_train, y_val, ids_train, ids_val, images_train, images_val = train_test_split(
    X_combined, y, property_ids, X_images,
    test_size=0.2, random_state=42
)

print(f"Training set (80% of training data):")
print(f"  - Features: {X_train.shape}")
print(f"  - Images: {images_train.shape}")
print(f"  - Target: {y_train.shape}")

print(f"\nValidation set (20% of training data):")
print(f"  - Features: {X_val.shape}")
print(f"  - Images: {images_val.shape}")
print(f"  - Target: {y_val.shape}")

print(f"\nTest set (separate test data - NO SPLIT):")
print(f"  - Features: {X_combined_test.shape}")
print(f"  - Images: {X_images_test.shape}")
print(f"  - Target: NOT AVAILABLE (to be predicted)")

# Check price distribution in splits
print(f"\nPrice distribution:")
print(f"Training - Mean: ${y_train.mean():,.2f}, Std: ${y_train.std():,.2f}")
print(f"Validation - Mean: ${y_val.mean():,.2f}, Std: ${y_val.std():,.2f}")

# ============================================================================
# 10. SAVE PREPROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("SAVING PREPROCESSED DATA")
print("="*80)

# Create directory for processed data
os.makedirs('preprocessed_data', exist_ok=True)

# Save TRAINING split numpy arrays
np.save('preprocessed_data/X_train.npy', X_train.values)
np.save('preprocessed_data/X_val.npy', X_val.values)
np.save('preprocessed_data/y_train.npy', y_train)
np.save('preprocessed_data/y_val.npy', y_val)
np.save('preprocessed_data/images_train.npy', images_train)
np.save('preprocessed_data/images_val.npy', images_val)
np.save('preprocessed_data/ids_train.npy', ids_train)
np.save('preprocessed_data/ids_val.npy', ids_val)

# Save TEST data numpy arrays
np.save('preprocessed_data/X_test.npy', X_combined_test.values)
np.save('preprocessed_data/images_test.npy', X_images_test)
np.save('preprocessed_data/ids_test.npy', test_property_ids)

print("Training, validation, and test data saved!")

# Save feature names and scaler
import pickle

with open('preprocessed_data/feature_names.pkl', 'wb') as f:
    pickle.dump(X_combined.columns.tolist(), f)

with open('preprocessed_data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('preprocessed_data/tabular_features.pkl', 'wb') as f:
    pickle.dump(tabular_features, f)

# Save metadata
metadata = {
    'n_train_samples': len(X_combined),
    'n_features': X_combined.shape[1],
    'n_tabular_features': len(tabular_features),
    'n_image_features': image_features_df.shape[1],
    'img_size': IMG_SIZE,
    'train_size': len(X_train),
    'val_size': len(X_val),
    'test_size': len(X_combined_test),
    'price_mean': y_train.mean(),
    'price_std': y_train.std()
}

with open('preprocessed_data/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("All preprocessed data saved successfully!")
print("\nSaved files:")
print("  TRAINING SPLIT:")
print("    - X_train.npy, X_val.npy")
print("    - y_train.npy, y_val.npy")
print("    - images_train.npy, images_val.npy")
print("    - ids_train.npy, ids_val.npy")
print("  TEST DATA:")
print("    - X_test.npy")
print("    - images_test.npy")
print("    - ids_test.npy")
print("  METADATA:")
print("    - feature_names.pkl, scaler.pkl, tabular_features.pkl, metadata.pkl")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print("Ready for model training and prediction!")