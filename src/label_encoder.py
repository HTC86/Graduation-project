import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# 加载数据集
data = np.load('shl_balanced_motion_location.npz')
y_train = data['y_train']

# 拟合标签编码器
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# 保存
output_file = 'label_encoder.pkl'
joblib.dump(label_encoder, output_file)
print(f"标签编码器已保存至 {output_file}")
print("类别顺序:", label_encoder.classes_)