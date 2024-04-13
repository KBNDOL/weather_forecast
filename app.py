from flask import Flask, request, render_template
import os
import tempfile
from werkzeug.utils import secure_filename
from forecast.forecast_temp import predict_temperature_from_excel
from forecast.forecast_pres import predict_pres_from_excel

app = Flask(__name__)

UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 执行温度预测
            try:
                predicted_temp = predict_temperature_from_excel(filepath)
                predicted_pressure = predict_pres_from_excel(filepath)
                prediction_text = f"Predicted Temperature: {predicted_temp:.2f}, " \
                                  f"Pressure: {predicted_pressure:.2f}, "
            except ValueError as e:
                prediction_text = str(e)

            return render_template('prediction.html', prediction=prediction_text)

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)

