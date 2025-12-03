import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from feature_extractor import extract_features, features_to_vector, FEATURE_ORDER
from tqdm import tqdm
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置 ---
# 线程数量：根据你的网速调整，通常 20-50 比较合适
# 太高可能会被网站封 IP，太低速度提不上来
MAX_WORKERS = 50 

# 定义路径
BASE_DIR = os.path.dirname(__file__)
NEW_DATA_PATH = os.path.join(BASE_DIR, "../data/phishing_dataset.csv")
MODEL_OUT = os.path.join(BASE_DIR, "../models/phishing_rf.pkl")

def process_single_url(data):
    """
    单个 URL 的处理函数，用于多线程调用
    """
    url, label = data
    try:
        # 这里调用你原来的提取逻辑，包含网络请求
        feats = extract_features(url)
        vec = features_to_vector(feats)
        return vec, label
    except Exception:
        # 如果某个 URL 提取失败（比如网站挂了），返回 None
        return None

def load_data_parallel(path):
    print(f"Loading raw data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    
    df = pd.read_csv(path)
    
    # 为了演示速度，如果你只想测试，可以取消下面这行的注释，只取前 1000 条
    # df = df.head(1000) 
    
    # 查找标签列
    target_col = 'status' if 'status' in df.columns else 'label'
    if 'url' not in df.columns or not target_col:
        raise ValueError("CSV must have 'url' and 'label' columns")

    urls = df['url'].values
    labels = df[target_col].values
    
    # 准备数据对
    data_pairs = list(zip(urls, labels))
    total = len(data_pairs)
    
    print(f"🚀 启动多线程提取 (线程数: {MAX_WORKERS})...")
    print("这会比单线程快几十倍，但仍需一点时间，请耐心等待...")
    
    processed_rows = []
    processed_labels = []
    
    start_time = time.time()
    
    # --- 核心：多线程并行执行 ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_url, pair): pair for pair in data_pairs}
        
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=total, unit="url"):
            result = future.result()
            if result is not None:
                vec, label = result
                processed_rows.append(vec)
                processed_labels.append(label)
                
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n✅ 特征提取完成！")
    print(f"耗时: {duration:.2f} 秒")
    print(f"平均速度: {len(processed_rows) / duration:.2f} URL/s")
    
    # 转换为 DataFrame
    X = pd.DataFrame(processed_rows, columns=FEATURE_ORDER)
    
    # 处理标签
    y_raw = pd.Series(processed_labels)
    if y_raw.dtype == object:
        y = y_raw.apply(lambda x: 1 if str(x).lower().strip() == 'phishing' else 0)
    else:
        y = y_raw.astype(int)
        
    return X, y

def main():
    # 1. 并行加载数据
    try:
        X, y = load_data_parallel(NEW_DATA_PATH)
    except Exception as e:
        print(f"Error: {e}")
        return

    if len(X) == 0:
        print("没有提取到有效数据，程序结束。")
        return

    # 2. 分割数据
    print(f"Splitting {len(X)} samples...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3. GPU 训练 (这步是毫秒级的)
    print("🚀 Training XGBoost with RTX 4070 Super...")
    
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        n_jobs=-1,
        device="cuda",      # 使用 GPU
        tree_method="hist"  # 极速模式
    )
    
    clf.fit(X_train, y_train)
    print("✅ Model trained!")

    # 4. 评估
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # 5. 保存
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump({'model': clf, 'feature_order': FEATURE_ORDER}, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()