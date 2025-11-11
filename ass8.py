import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Load and preprocess ===
df = pd.read_csv(rf"C:\Users\ab\Downloads\RS-A5_amazon_products_sales_data_cleaned.csv")
df['product_rating'] = df['product_rating'].fillna(df['product_rating'].mean())
df = df.sample(5000, random_state=42).reset_index(drop=True)

# Add dummy user and product IDs
df['user_id'] = np.random.randint(0, 500, len(df))  # 500 users for better overlap
df['product_id'] = df.index

# === 2. Create user-item matrix ===
pivot = df.pivot_table(index='user_id', columns='product_id', values='product_rating').fillna(0)

# === 3. Apply SVD ===
svd = TruncatedSVD(n_components=50, random_state=42)
latent_matrix = svd.fit_transform(pivot)
reconstructed = np.dot(latent_matrix, svd.components_)

# === 4. Recommend top 5 items for a random user ===
# NOTE: train was undefined — we should pick from pivot.index instead
user_id = np.random.choice(pivot.index)
user_idx = pivot.index.get_loc(user_id)

pred_ratings = reconstructed[user_idx]

# Find items already rated by the user
rated_items = pivot.columns[pivot.iloc[user_idx] > 0]

# Exclude already-rated items from recommendations
pred_ratings = {pid: score for pid, score in zip(pivot.columns, pred_ratings) if pid not in rated_items}

# Sort predicted ratings
top5 = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:5]

# === 5. Display results ===
print(f"\nTop 5 Recommendations for user ID {user_id} :\n")
for i,_ in top5:
    print(i, "--->", df.loc[i, 'product_title'])