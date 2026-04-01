from flask import Flask, render_template, request, redirect, session, url_for, flash, send_file
import sqlite3
import joblib
import csv
from io import StringIO
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
from send_mail import send_stress_email
import smtplib

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SESSION_PERMANENT'] = False
app.permanent_session_lifetime = timedelta(minutes=30)

# Load vectorizer for tweets
vectorizer_tweets = joblib.load("models/tfidf_vectorizer_tweets.pkl")

# Models for tweets
models_tweets = {
    'mlp': joblib.load("models/mlp_tweets.pkl"),
    'svm': joblib.load("models/svm_tweets.pkl"),
    'naive_bayes': joblib.load("models/naive_bayes_tweets.pkl"),
    'decision_tree': joblib.load("models/decision_tree_tweets.pkl"),
    'logistic_regression': joblib.load("models/logistic_regression_tweets.pkl"),
    'gradient_boosting': joblib.load("models/gradient_boosting_tweets.pkl"),
    'random_forest': joblib.load("models/random_forest_tweets.pkl")
}

# Models for depression questionnaire
models_depression = {
    'mlp': joblib.load("models/mlp_depression.pkl"),
    'svm': joblib.load("models/svm_depression.pkl"),
    'naive_bayes': joblib.load("models/naive_bayes_depression.pkl"),
    'decision_tree': joblib.load("models/decision_tree_depression.pkl"),
    'logistic_regression': joblib.load("models/logistic_regression_depression.pkl"),
    'gradient_boosting': joblib.load("models/gradient_boosting_depression.pkl"),
    'random_forest': joblib.load("models/random_forest_depression.pkl")
}

# Initialize the database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    admin INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    model TEXT,
                    result TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('mainhome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        confirm = request.form['confirm']

        if not (name and username and email and password and confirm):
            flash('Please fill all fields', 'danger')
            return render_template('register.html')

        if password != confirm:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (name, username, email, password) VALUES (?, ?, ?, ?)",
                      (name, username, email, hashed_password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError as e:
            if 'username' in str(e).lower():
                flash('Username already exists.', 'danger')
            elif 'email' in str(e).lower():
                flash('Email already registered.', 'danger')
            else:
                flash('Database error occurred.', 'danger')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        if username == 'admin' and password == 'admin':
            session['username'] = 'admin'
            session['admin'] = True
            return redirect(url_for('admin'))

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            session['username'] = username
            session['admin'] = False
            flash('Login successful', 'success')
            return redirect(url_for('mainhome'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/mainhome')
def mainhome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('mainhome.html', username=session['username'])

@app.route('/about')
def about():
    return render_template('aboutpage.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    result_message = None
    if request.method == 'POST':
        input_type = request.form.get('input_type')
        model_choice = request.form.get('model')

        try:
            if input_type == 'tweets':
                message = request.form.get('message', '').strip()
                if not message:
                    flash('Please enter the tweet message.', 'danger')
                    return render_template('home.html', username=session['username'])

                features = vectorizer_tweets.transform([message])
                model = models_tweets.get(model_choice)
                if not model:
                    flash('Invalid model selected.', 'danger')
                    return render_template('home.html', username=session['username'])

                prediction = model.predict(features)[0]
                result_message = "Stressed/Depressed" if prediction == 1 else "Normal"

            else:
                questions = []
                for i in range(1, 11):
                    val = request.form.get(f'q{i}')
                    if val is None or val.strip() == '':
                        flash(f'Please answer question {i}.', 'danger')
                        return render_template('home.html', username=session['username'])
                    questions.append(float(val))

                score = request.form.get('score')
                questions.append(float(score) if score and score.strip() != '' else 0.0)

                model = models_depression.get(model_choice)
                if not model:
                    flash('Invalid model selected.', 'danger')
                    return render_template('home.html', username=session['username'])

                prediction = model.predict([questions])[0]
                class_labels = {
                    0: "No Depression",
                    1: "Mild Depression",
                    2: "Moderate Depression",
                    3: "Severe Depression"
                }
                result_message = class_labels.get(prediction, "Unknown")
                message = ','.join(map(str, questions))

            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute(
                "INSERT INTO results (username, input_type, message, model, result) VALUES (?, ?, ?, ?, ?)",
                (session['username'], input_type, message, model_choice, result_message)
            )
            conn.commit()
            conn.close()

        except Exception as e:
            print("Error during prediction:", e)
            flash("Invalid input or model error.", "danger")

    return render_template('home.html', username=session['username'], result=result_message)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'username' not in session or session.get('username') != 'admin':
        return redirect(url_for('login'))

    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if request.method == 'POST':
        if 'clear' in request.form:
            c.execute("DELETE FROM results")
            conn.commit()
            flash('All results cleared.', 'success')
        elif 'download' in request.form:
            c.execute("SELECT * FROM results")
            data = c.fetchall()
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(['ID', 'Username', 'Input Type', 'Message', 'Model', 'Result'])
            for row in data:
                writer.writerow([row['id'], row['username'], row['input_type'], row['message'], row['model'], row['result']])
            output.seek(0)
            conn.close()
            return send_file(output, mimetype='text/csv', download_name='report.csv', as_attachment=True)

    c.execute("SELECT * FROM results")
    data = c.fetchall()
    conn.close()
    return render_template('admin.html', data=data)
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'username' not in session or not session.get('username') == 'admin':
        return redirect('/login')

    filter_value = request.args.get('filter', 'All')
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if filter_value == 'All':
        cursor.execute("""
            SELECT posts.id, users.username, posts.input_type, 
                   posts.content as message, posts.predictions as result 
            FROM posts 
            JOIN users ON posts.user_id = users.id
        """)
    elif filter_value == 'Normal':
        cursor.execute("""
            SELECT posts.id, users.username, posts.input_type, 
                   posts.content as message, posts.predictions as result 
            FROM posts 
            JOIN users ON posts.user_id = users.id 
            WHERE posts.predictions NOT IN (
                'Stressed/Depressed', 'Mild Depression', 
                'Moderate Depression', 'Severe Depression'
            )
        """)
    else:
        cursor.execute("""
            SELECT posts.id, users.username, posts.input_type, 
                   posts.content as message, posts.predictions as result 
            FROM posts 
            JOIN users ON posts.user_id = users.id 
            WHERE posts.predictions = ?
        """, (filter_value,))

    data = cursor.fetchall()
    conn.close()

    return render_template('admin.html', data=data, filter=filter_value)


@app.route('/preview_report/<int:id>', methods=['GET', 'POST'])
def preview_report(id=None):
    # you can ignore `id` if you don't need it
    if 'username' not in session or session.get('username') != 'admin':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT users.name, users.email, results.message, results.result
        FROM results
        JOIN users ON results.username = users.username
        ORDER BY results.id DESC
    ''')
    data = c.fetchall()
    conn.close()

    return render_template('report.html', data=data)


@app.route('/send_email', methods=['POST'])
def send_email():
    if 'username' not in session or session.get('username') != 'admin':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    email = request.form['email']
    name = request.form['name']
    message = request.form['message']
    result = request.form['result']

    success = send_stress_email(email, name, message, result)

    if success:
        flash('Email sent successfully!', 'success')
        return redirect(url_for('email_sent'))
    else:
        flash('Failed to send email.', 'danger')
        return redirect(url_for('preview_report'))

@app.route('/email_sent')
def email_sent():
    if 'username' not in session or session.get('username') != 'admin':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    return render_template('email_sent.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
