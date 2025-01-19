import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from NeuralModel import NeuralModelForPP
from Osu.OsuData import OsuData
from Osu.OsuUser import OsuUser


osudata = OsuData()
dataset = osudata.get_all_users_from_db()

X = []
y = []

for user in dataset:
    X.append([user.hit_accuracy, user.play_count, user.play_time])  
    y.append([user.pp]) 
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Размер обучающей выборки X_train: {X_train.shape}")
print(f"Размер тестовой выборки X_test: {X_test.shape}")
print(f"Размер обучающей выборки y_train: {y_train.shape}")
print(f"Размер тестовой выборки y_test: {y_test.shape}")

models = [NeuralModelForPP('relu',X_unscaled=X_train,loss_function_name="MAE",epochs=5000)]
for model in models:
    model.train(X_train_scaled, y_train)
    print("test")
    model.evaluate(X_test_scaled,y_test,unscaled_data=X_test)