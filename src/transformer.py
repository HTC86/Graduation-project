import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==================== 加载数据集 ====================
data = np.load('shl_balanced_motion_location.npz')
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

# ==================== 标签编码 ====================
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)
class_names = label_encoder.classes_
num_classes = len(class_names)

# ==================== 数据标准化 ====================
n_samples, time_steps, n_features = X_train.shape

X_train_2d = X_train.reshape(-1, n_features)
scaler = StandardScaler()
scaler.fit(X_train_2d)

def scale_data(X, scaler):
    original_shape = X.shape
    X_2d = X.reshape(-1, n_features)
    X_scaled_2d = scaler.transform(X_2d)
    return X_scaled_2d.reshape(original_shape).astype(np.float32)

X_train = scale_data(X_train, scaler)
X_val   = scale_data(X_val, scaler)
X_test  = scale_data(X_test, scaler)

joblib.dump(scaler, 'scaler_informer.pkl')

# ==================== 位置编码 ====================
def positional_encoding(length, d_model):
    angle_rads = get_angles(np.arange(length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

# ==================== ProbSparseSelfAttention====================
class ProbSparseSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads, factor=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.factor = factor
        self.dropout = dropout
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        batch, time, d_model = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        # 安全处理 training 参数
        if training is None:
            is_training = tf.constant(False)
        else:
            is_training = tf.convert_to_tensor(training, dtype=tf.bool)

        def sparse_attention():
            sample_size = tf.minimum(self.factor, time)
            indices = tf.random.shuffle(tf.range(time))[:sample_size]
            q_sampled = tf.gather(x, indices, axis=1)
            return self.mha(query=x, value=x, key=x, training=training)

        def full_attention():
            return self.mha(query=x, value=x, key=x, training=training)

        attn_output = tf.cond(
            tf.logical_and(is_training, tf.greater(time, self.factor)),
            sparse_attention,
            full_attention
        )
        out = self.layer_norm(x + attn_output)
        return out

# ==================== TransformerEncoder ====================
class TransformerEncoder(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, use_sparse=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_sparse = use_sparse

        if use_sparse:
            self.attention = ProbSparseSelfAttention(d_model, num_heads, factor=10, dropout=dropout)
        else:
            self.attention = layers.MultiHeadAttention(num_heads, d_model // num_heads, dropout=dropout)
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ])
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        if self.use_sparse:
            attn_output = self.attention(x, training=training)
        else:
            attn_output = self.attention(query=x, value=x, key=x, training=training)
            attn_output = self.norm1(x + attn_output)
            x = attn_output

        ffn_output = self.ffn(x, training=training)
        out = self.norm2(x + ffn_output)
        return out

# ==================== 构建 Informer 模型 ====================
def build_informer(input_shape, num_classes, d_model=64, num_heads=4, ff_dim=128,
                   num_layers=3, dropout=0.1, use_sparse=True):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(d_model, activation='relu')(inputs)
    pos_enc = positional_encoding(input_shape[0], d_model)
    x = x + pos_enc
    x = layers.Dropout(dropout)(x)

    for _ in range(num_layers):
        x = TransformerEncoder(d_model, num_heads, ff_dim, dropout, use_sparse)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# ==================== 创建模型 ====================
input_shape = (time_steps, n_features)
model = build_informer(input_shape, num_classes,
                       d_model=64,
                       num_heads=4,
                       ff_dim=128,
                       num_layers=3,
                       dropout=0.1,
                       use_sparse=True)
model.summary()

# ==================== 编译与训练 ====================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 数据增强
def add_noise(x, y, noise_factor=0.02):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_factor, dtype=x.dtype)
    return x + noise, y

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_enc))
train_dataset = train_dataset.shuffle(1024).batch(batch_size) \
    .map(lambda x, y: add_noise(x, y, noise_factor=0.02),
         num_parallel_calls=tf.data.AUTOTUNE) \
    .prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_enc)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_enc)).batch(batch_size)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('informer_best.keras', monitor='val_accuracy', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

print("\n开始训练 Informer ...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

# ==================== 评估 ====================
model.load_weights('informer_best.keras')
loss, acc = model.evaluate(test_dataset, verbose=0)
print(f"\nInformer 测试准确率: {acc:.4f}")

y_pred_prob = model.predict(test_dataset)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n分类报告:")
print(classification_report(y_test_enc, y_pred, target_names=class_names))

cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Informer 混淆矩阵')
plt.xlabel('预测')
plt.ylabel('真实')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['loss'], label='train')
ax1.plot(history.history['val_loss'], label='val')
ax1.set_title('Loss')
ax1.legend()
ax2.plot(history.history['accuracy'], label='train')
ax2.plot(history.history['val_accuracy'], label='val')
ax2.set_title('Accuracy')
ax2.legend()
plt.show()