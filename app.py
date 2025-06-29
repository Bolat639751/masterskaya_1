from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import io

app = FastAPI()

model_path = r"C:\Users\rank-\OneDrive\Рабочий стол\Проекты аналитика\Мастерская\model.pkl"
model = joblib.load(model_path)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Загрузка файла для предсказаний</title>
        </head>
        <body>
            <h2>Загрузите CSV файл</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".csv" required>
                <button type="submit">Отправить</button>
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode('utf-8')))

    # Обработка данных (ваш код)
    df = df.drop(['Unnamed: 0', 'Family History'], axis=1)

    # Обработка названий колонок
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace('(', '', regex=False)
    df.columns = df.columns.str.replace(')', '', regex=False)
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    df.columns = df.columns.str.replace('-', '_', regex=False)
    df.columns = df.columns.str.replace('_binary', '', regex=False)

    # Предсказание вероятностей
    proba_values = model.predict_proba(df.drop(columns=['id']))[:, 1]

    # Создаем DataFrame с результатами
    result_df = df[['id']].copy()
    result_df['pred_proba'] = proba_values
    result_df['prediction'] = result_df['pred_proba'].apply(lambda x: 0 if x >= 0.42 else 1)

    # Формируем HTML таблицу с результатами:
    html_content = """
    <html>
        <head>
            <title>Результаты предсказаний</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h2>Результаты предсказаний для файла "{filename}"</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Вероятность</th>
                    <th>Предсказание</th>
                </tr>"""

    for _, row in result_df.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['id']}</td>
                    <td>{row['pred_proba']:.4f}</td>
                    <td>{int(row['prediction'])}</td>
                </tr>"""

    html_content += """
            </table>
            <br><a href="/">Загрузить другой файл</a>
        </body>
    </html>"""

    return HTMLResponse(content=html_content)