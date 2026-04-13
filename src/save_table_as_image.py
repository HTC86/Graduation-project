import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table

# 读取 CSV 文件
df = pd.read_csv('cv_results/cv_summary.csv')

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis('off')

# 创建表格
table = Table(ax, bbox=[0, 0, 1, 1])
n_rows, n_cols = df.shape
n_cols += 1

# 设置列宽和行高
col_widths = [0.05] + [0.18] * (n_cols - 1)
row_height = 0.15

# 添加表头
headers = ['#'] + list(df.columns)
for j, header in enumerate(headers):
    table.add_cell(0, j, width=col_widths[j], height=row_height, text=header,
                   loc='center', facecolor='#40466e', edgecolor='white')
    table.get_celld()[(0, j)].get_text().set_color('white')
    table.get_celld()[(0, j)].get_text().set_weight('bold')

# 添加数据行
for i in range(n_rows):
    # 行号
    table.add_cell(i+1, 0, width=col_widths[0], height=row_height, text=str(i+1),
                   loc='center', facecolor='#f0f0f0', edgecolor='lightgray')
    for j in range(n_cols-1):
        value = df.iloc[i, j]
        # 数值格式化：保留4位小数
        if isinstance(value, float):
            text = f"{value:.4f}"
        else:
            text = str(value)
        table.add_cell(i+1, j+1, width=col_widths[j+1], height=row_height, text=text,
                       loc='center', facecolor='white', edgecolor='lightgray')

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)
ax.add_table(table)

plt.title('5-Fold Cross-Validation Summary', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('cv_summary_table.png', dpi=200, bbox_inches='tight')
plt.show()
print("表格图片已保存为 cv_summary_table.png")