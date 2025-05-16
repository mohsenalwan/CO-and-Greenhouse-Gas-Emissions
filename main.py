import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# إنشاء المجلدات
os.makedirs("Result", exist_ok=True)
os.makedirs("TrainTestSplit", exist_ok=True)

# تحميل البيانات
df = pd.read_csv("Data/co2_emission.csv")

# تنظيف البيانات
df.columns = df.columns.str.strip().str.lower()
df.rename(columns={'annual co₂ emissions (tonnes )': 'co2'}, inplace=True)
df.dropna(subset=['co2'], inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# خصائص ومدخلات
X = df[['year']]
y = df['co2']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# حفظ ملفات Train/Test
X_train.to_csv("TrainTestSplit/X_train.csv", index=False)
X_test.to_csv("TrainTestSplit/X_test.csv", index=False)
y_train.to_csv("TrainTestSplit/y_train.csv", index=False)
y_test.to_csv("TrainTestSplit/y_test.csv", index=False)

# حفظ test_set الكامل في Result
test_set = pd.concat([X_test, y_test], axis=1)
test_set.to_csv("Result/test_set.csv", index=False)

# تدريب النماذج
models = {
    'lr': LinearRegression(),
    'rf': RandomForestRegressor(random_state=42),
    'svr': SVR(),
    'knn': KNeighborsRegressor(),
    'tree': DecisionTreeRegressor(random_state=42)
}

# تنبؤ وتخزين النتائج
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_df = pd.DataFrame({
        'year': X_test['year'],
        'actual': y_test,
        'predicted': preds
    })
    preds_df.to_csv(f"Result/predictions_{name}.csv", index=False)

print("✅ تم إنشاء جميع الملفات المطلوبة في المجلدين Result/ و TrainTestSplit/")
