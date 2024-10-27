import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from dash import Dash, dcc, html
import plotly.express as px

# 1. 데이터 생성
np.random.seed(42)
data = pd.DataFrame({
    'customer_id': np.arange(1, 101),
    'age': np.random.randint(18, 70, size=100),
    'annual_income': np.random.normal(50000, 15000, 100).astype(int),
    'purchase_amount': np.random.normal(1000, 300, 100).astype(int),
    'latitude': np.random.uniform(35.0, 38.0, 100),
    'longitude': np.random.uniform(125.0, 129.0, 100),
    'loyalty_score': np.random.randint(1, 5, 100)
})

# 2. 데이터 정리 및 EDA
plt.figure(figsize=(10, 6))
sns.histplot(data['annual_income'], kde=True)
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 3. SQL을 이용한 데이터 필터링
conn = sqlite3.connect('customer_data.db')
data.to_sql('customer_data', conn, if_exists='replace', index=False)
query = 'SELECT age, AVG(purchase_amount) as avg_purchase FROM customer_data GROUP BY age'
sql_data = pd.read_sql_query(query, conn)

plt.figure(figsize=(12, 6))
sns.lineplot(data=sql_data, x='age', y='avg_purchase')
plt.title('Average Purchase by Age')
plt.xlabel('Age')
plt.ylabel('Average Purchase Amount')
plt.show()

# 4. Folium 대화형 지도 시각화
m = folium.Map(location=[36.5, 127.0], zoom_start=7)
for idx, row in data.iterrows():
    folium.Marker([row['latitude'], row['longitude']],
                  popup=f"Customer ID: {row['customer_id']}, Purchase: ${row['purchase_amount']}").add_to(m)
m.save('customer_map.html')
print("대화형 지도가 'customer_map.html'로 저장되었습니다.")

# 5. 예측 분석 (분류 모델)
X = data[['age', 'annual_income', 'purchase_amount']]
y = data['loyalty_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("분류 모델 성능 평가:\n", classification_report(y_test, y_pred))

# 6. Plotly Dash 대시보드
app = Dash(__name__)
fig = px.scatter(data, x='annual_income', y='purchase_amount', color='loyalty_score', title='Annual Income vs Purchase Amount by Loyalty Score')
app.layout = html.Div([
    html.H1("Customer Data Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    print("Dash 대시보드를 실행하려면 http://127.0.0.1:8050 에 접속하세요.")
    app.run_server(debug=True)

# 7. 결론 요약 저장
with open("summary.txt", "w") as file:
    file.write("프로젝트 주요 분석 결과 요약:\n")
    file.write("1. 연령대에 따른 구매 패턴 분석\n")
    file.write("2. 연간 소득과 구매 금액의 상관 관계 파악\n")
    file.write("3. 로열티 점수 예측을 위한 분류 모델 성능 평가\n")
print("요약 결과가 'summary.txt'에 저장되었습니다.")
