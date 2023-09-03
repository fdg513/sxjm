import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

def rotate_point(x, y, angle):
    # 将点(x, y)绕原点逆时针旋转angle弧度
    x_new = x * math.cos(angle) - y * math.sin(angle)
    y_new = x * math.sin(angle) + y * math.cos(angle)
    return x_new, y_new

def calculate_coverage_area(width, length, rotation_angle, translation_distance, num_samples):
    area = 0.0

    for _ in range(num_samples):
        x = np.random.uniform(-width / 2, width / 2)
        y = np.random.uniform(-length / 2, length / 2)

        # 计算旋转后的坐标
        x_rotated, y_rotated = rotate_point(x, y, rotation_angle)

        # 平移
        x_translated = x_rotated + translation_distance

        # 检查点是否在长方形内
        if -width / 2 <= x_translated <= width / 2 and -length / 2 <= y_rotated <= length / 2:
            area += 1

    # 计算覆盖面积的估计值
    estimated_area = area * (width * length) / num_samples
    return estimated_area

# 固定其他参数
width = 2.0
length = 1.0
translation_distance = 1.0
num_samples = 1000000

# 计算不同旋转角度下的扫过面积
rotation_angles_degrees = np.arange(0, 91, 1)  # 从0度到90度，每隔1度
rotation_angles = np.radians(rotation_angles_degrees)
estimated_areas = []

for rotation_angle in rotation_angles:
    estimated_area = calculate_coverage_area(width, length, rotation_angle, translation_distance, num_samples)
    estimated_areas.append(estimated_area)

# 绘制旋转角度与扫过面积的关系
plt.figure(figsize=(10, 6))
plt.plot(rotation_angles_degrees, estimated_areas, marker='o')
plt.xlabel('Rotation Angle (Degrees)')
plt.ylabel('Estimated Area')
plt.title('Estimated Area vs. Rotation Angle (Fixed Parameters)')
plt.grid(True)
plt.show()

# 使用线性回归拟合一个函数来描述估计的面积与旋转角度之间的关系
rotation_angles_degrees = rotation_angles_degrees.reshape(-1, 1)
model = LinearRegression()
model.fit(rotation_angles_degrees, estimated_areas)

# 打印线性回归模型的系数
print("Linear Regression Coefficients:")
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# 使用神经网络回归拟合一个函数
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp_regressor.fit(rotation_angles_degrees, estimated_areas)

# 打印神经网络回归模型的R²得分
r_squared = mlp_regressor.score(rotation_angles_degrees, estimated_areas)
print("R-squared (Neural Network):", r_squared)
