import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def correlation_filter(X, y, threshold=0.05):
    n_features = X.shape[1]
    # 计算每个特征与y的皮尔逊相关系数
    corr_list = []
    for i in range(n_features):
        corr, _ = pearsonr(X[:, i], y)
        corr_list.append(corr)
    corr_array = np.array(corr_list)
    
    # 筛选相关系数绝对值 >= threshold 的特征
    selected_idx = np.where(np.abs(corr_array) >= threshold)[0]
    print(f"相关系数过滤法: 原始特征数={n_features}, 保留特征数={len(selected_idx)}, 阈值={threshold}")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    plt.bar(range(n_features), np.abs(corr_array))
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'threshold={threshold}')
    plt.xlabel('特征索引')
    plt.ylabel('|皮尔逊相关系数|')
    plt.title('特征与目标的相关性强度')
    plt.legend()
    plt.show()
    
    return selected_idx

def tree_importance_filter(X, y, cumulative_importance=0.95, random_state=42):
    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    # 按重要性降序排序
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    
    # 计算累积重要性，确定要保留的特征数
    cumsum = np.cumsum(sorted_importances)
    n_selected = np.searchsorted(cumsum, cumulative_importance) + 1
    selected_idx = indices[:n_selected]
    
    print(f"随机森林特征重要性法: 原始特征数={X.shape[1]}, 保留特征数={n_selected}, 累积重要性={cumulative_importance}")
    print(f"保留特征对应的累计重要性: {cumsum[n_selected-1]:.4f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 5))
    plt.bar(range(n_selected), sorted_importances[:n_selected])
    plt.axhline(y=sorted_importances[n_selected-1], color='r', linestyle='--', label='入选阈值')
    plt.xlabel('特征排名')
    plt.ylabel('特征重要性')
    plt.title(f'随机森林特征重要性（保留前{n_selected}个特征）')
    plt.legend()
    plt.show()
    
    return selected_idx