"""Module with all the view (route) functions."""

from ast import literal_eval
import os

from flask import redirect, render_template, request, session
from pandas.api.types import is_string_dtype, is_numeric_dtype
from werkzeug.utils import secure_filename

from . import __version__, app, processing


@app.route('/', methods=['GET', 'POST'])
def index():
    files = processing.find_files()
    if 'URL' in app.config: files.update(app.config['URL'])
    if request.method == 'POST':
        file = request.files['file']
        if file and processing._allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(os.path.dirname(__file__), filename))
            return redirect(request.url)
    return render_template('index.html', version=__version__, files=files)


@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        session['file'] = request.form['file']
        session.modified = True
        name, path = literal_eval(session['file'].replace('\\', '/'))
        features = processing.get_xy(path)[0]
        categorical = {f: features[f].unique() for f in features
                       if is_string_dtype(features[f])}
        numerical = [f for f in features if is_numeric_dtype(features[f])]
    return render_template('process.html', version=__version__,
                           name=name, path=path,
                           categorical=categorical, numerical=numerical)


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        session['form'] = dict(request.form.items())
        session.modified = True
        name = session['form'].pop('name', None)
        path = session['form'].pop('path', None)
        features, outcome = processing.get_xy(path)
        categorical = {f: features[f].unique() for f in features
                       if is_string_dtype(features[f])}
        category = ('').join(
            [v for k, v in session['form'].items() if k in categorical])
        cf = features[categorical].sum(axis=1)
        if any([not value for value in session['form'].values()]):
            prediction = {'<font color="red">Error': 'Failed to collect data'}
        else:
            X, y, sample = processing.process_data(path, session['form'])
            if category:
                X = X.loc[(cf[cf == category]).index]
                y = y.loc[(cf[cf == category]).index]
            ols = processing.predict_ols(X, y, sample, len(X) - 1)
            prediction = {
                f'<strong>{ols["mean"]:.5g}</strong>':
                f'({ols["mean_ci_lower"]:.5g} ÷ {ols["mean_ci_upper"]:.5g})'
            }
            if not all(sample.squeeze(axis=0).between(X.min(), X.max())):
                prediction['<font color="orange">Warning'] = 'Out-of-sample'
    return render_template('result.html', version=__version__,
                           form=session['form'], name=name, path=path,
                           outcome=outcome.name, prediction=prediction)
