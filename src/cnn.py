import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 加载数据集
data = np.load('shl_balanced_motion_location.npz')
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

# 将标签编码为整数
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

class_names = label_encoder.classes_
num_classes = len(class_names)

# 标准化
n_samples_train, time_steps, n_features = X_train.shape
X_train_2d = X_train.reshape(-1, n_features)
scaler = StandardScaler()
scaler.fit(X_train_2d)


def scale_data(X, scaler):
    original_shape = X.shape
    X_2d = X.reshape(-1, n_features)
    X_scaled_2d = scaler.transform(X_2d)
    return X_scaled_2d.reshape(original_shape)

X_train = scale_data(X_train, scaler)
X_val = scale_data(X_val, scaler)
X_test = scale_data(X_test, scaler)

joblib.dump(scaler, 'scaler.pkl')


# 1D-CNN 模型
def create_1d_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (X_train.shape[1], X_train.shape[2])
print(f"\n输入形状: {input_shape}")
model_cnn = create_1d_cnn(input_shape, num_classes)
model_cnn.summary()


# 训练配置
def compile_and_train(model, X_train, y_train, X_val, y_val, model_name, epochs=100, batch_size=128):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(f'{model_name}.h5', monitor='val_accuracy', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history


# 训练
print("\n训练 1D-CNN ...")
history_cnn = compile_and_train(model_cnn, X_train, y_train_enc, X_val, y_val_enc, '1d_cnn')


# 评估
def evaluate_model(model, X_test, y_test_enc, class_names, model_name):
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
    print(f"\n{model_name} 测试准确率: {acc:.4f}")

    print("\n分类报告:")
    print(classification_report(y_test_enc, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test_enc, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.show()

    return y_pred

y_pred_cnn = evaluate_model(model_cnn, X_test, y_test_enc, class_names, '1D-CNN')


# 绘制训练曲线
def plot_training_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='train_loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train_acc')
    ax2.plot(history.history['val_accuracy'], label='val_acc')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.show()

plot_training_history(history_cnn, '1D-CNN')