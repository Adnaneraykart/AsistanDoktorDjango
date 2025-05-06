# ----------------------------------------------------------------------
#  AsistanDoktor – v4.5-no-env (Groq llama-3.3-70b-versatile) – FIX v5
# ----------------------------------------------------------------------
import re, json, sqlite3, logging
from datetime import datetime, timedelta
from queue import Queue
from functools import wraps

import groq, pandas as pd, joblib
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash

# ------------------------- Sabit Ayarlar -------------------------------
GROQ_API_KEY   = "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"          # ← gerçek anahtarınız
MODEL_NAME     = "llama-3.3-70b-versatile"
SECRET_KEY     = "change_this_in_prod"

client = groq.Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)
app.secret_key = SECRET_KEY
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)8s | %(message)s",
                    datefmt="%H:%M:%S")

BASE_DIR = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
DB_PATH  = f"{BASE_DIR}/users.db"

model         = joblib.load('xgb_model_multiclass.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# ------------------------ Semptom Sözlüğü ------------------------------
SYMPTOMS = {
    'age': {'type': 'number', 'default': 0},
    'gender': {'type': 'categorical', 'map': {'erkek': 1, 'kadın': 0}, 'default': 0},
    'blood_pressure': {'type': 'categorical',
                       'map': {'yüksek': 1, 'normal': 0, 'düşük': -1},
                       'default': 0},
    'cholesterol': {'type': 'categorical',
                    'map': {'yüksek': 1, 'normal': 0, 'düşük': -1},
                    'default': 0},
    'smoking': {'type': 'categorical', 'map': {'evet': 1, 'hayır': 0}, 'default': 0},
    'alcohol': {'type': 'categorical', 'map': {'evet': 1, 'hayır': 0}, 'default': 0},
    'stress': {'type': 'categorical', 'map': {'evet': 1, 'hayır': 0}, 'default': 0},
    'fever': {'type': 'categorical', 'map': {'evet': 1, 'hayır': 0}, 'default': 0},
    'cough': {'type': 'categorical', 'map': {'evet': 1, 'hayır': 0}, 'default': 0},
    'fatigue': {'type': 'categorical', 'map': {'evet': 1, 'hayır': 0}, 'default': 0},
    'difficulty_breathing': {'type': 'categorical',
                             'map': {'evet': 1, 'hayır': 0},
                             'default': 0},
}

# ------------------ Groq llama Semptom Çözümleyici ---------------------
SYS_PROMPT = (
    "You are a medical triage assistant.\n"
    "User text is in Turkish. Return ONLY a valid JSON object with keys:\n"
    "age, gender, blood_pressure, cholesterol, smoking, alcohol, stress, fever, "
    "cough, fatigue, difficulty_breathing.\n"
    "Values must be numeric: gender 1=male 0=female; categorical 1=yes 0=no -1=low; "
    "blood_pressure/cholesterol high=1 normal=0 low=-1. Use 0 if uncertain.\n"
    "Do NOT wrap the JSON in markdown or text."
)

def llama_json(txt: str) -> dict:
    logging.info("Groq‣Aldı : %s", txt)
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYS_PROMPT},
                      {"role": "user", "content": txt}],
            temperature=0)
        raw = r.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(m.group(0)) if m else {}
        logging.info("Groq‣Ham  : %s", parsed)
        return parsed
    except Exception as e:
        logging.exception("Groq JSON parse failed: %s", e)
        return {}

def extract_symptoms(txt: str) -> dict:
    p = llama_json(txt)
    EXTRA = {'gender': {'male': 1, 'female': 0},
             'blood_pressure': {'high': 1, 'normal': 0, 'low': -1},
             'cholesterol': {'high': 1, 'normal': 0, 'low': -1},
             'smoking': {'yes': 1, 'no': 0}, 'alcohol': {'yes': 1, 'no': 0},
             'stress': {'yes': 1, 'no': 0}, 'fever': {'yes': 1, 'no': 0},
             'cough': {'yes': 1, 'no': 0}, 'fatigue': {'yes': 1, 'no': 0},
             'difficulty_breathing': {'yes': 1, 'no': 0}}
    clean = {}
    for k, m in SYMPTOMS.items():
        v = p.get(k, m['default'])
        if m['type'] == 'number':
            try:
                v = float(v)
            except:  # type: ignore
                v = m['default']
            if k == 'age' and not 0 <= v <= 120:
                v = m['default']
        else:
            v = int(v) if isinstance(v, (int, float)) else \
                m['map'].get(str(v).lower()) or EXTRA[k].get(str(v).lower(), m['default'])
        clean[k] = v
    logging.info("Groq‣Temiz: %s", clean)
    return clean

# ------------------ LLM Tabanlı Öneri Üretimi --------------------------
RECOMMEND_PROMPT = (
    "You are a medical assistant. For each disease in the list below, "
    "return 2–3 short, patient-friendly Turkish bullet-point recommendations. "
    "Respond ONLY with JSON whose keys are the diseases and values are arrays.\n\n"
    "Diseases: {disease_list}"
)

def generate_recommendations(diseases: list[str]) -> dict:
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": RECOMMEND_PROMPT.format(
                disease_list=", ".join(diseases))}],
            temperature=0.3)
        raw = r.choices[0].message.content.strip()
        try:
            recs = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            recs = json.loads(m.group(0)) if m else {}
    except Exception as e:
        logging.exception("Groq recommendation failed: %s", e)
        recs = {}
    generic = ["Genel öneri: Bir sağlık uzmanına danışın."]
    return {d: " • ".join(recs.get(d, generic))
            if isinstance(recs.get(d, generic), list) else str(recs.get(d, generic))
            for d in diseases}

# ------------------ DB & Auth Yardımcıları -----------------------------
event_q = Queue()
def get_db():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'user'
        );
        CREATE TABLE IF NOT EXISTS logins (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            ts TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            disease TEXT,
            prob REAL,
            ts TEXT
        );
        """)
        if not c.execute("SELECT 1 FROM users WHERE username='admin'").fetchone():
            c.execute("INSERT INTO users(username,password,role) VALUES(?,?,?)",
                      ('admin', generate_password_hash('admin123'), 'admin'))

def register_user(u, p):
    with get_db() as c:
        c.execute("INSERT INTO users(username,password,role) VALUES(?,?,?)",
                  (u, generate_password_hash(p), 'user'))

def check_user(u, p):
    with get_db() as c:
        row = c.execute("SELECT password,role,id FROM users WHERE username=?", (u,)).fetchone()
    return {'role': row[1], 'id': row[2]} if row and check_password_hash(row[0], p) else None

def login_required(f):
    @wraps(f)
    def w(*a, **kw):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*a, **kw)
    return w

def admin_required(f):
    @wraps(f)
    def w(*a, **kw):
        if not session.get('is_admin'):
            flash("Yetkisiz erişim!", "danger")
            return redirect(url_for('home'))
        return f(*a, **kw)
    return w

# ---------------------------- Rotalar ----------------------------------
@app.route('/')
def idx():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        u, p = request.form.get('username'), request.form.get('password')
        if not u or not p:
            flash("Boş alan bırakmayın", "danger")
        else:
            try:
                register_user(u, p)
                flash("Kayıt başarılı!", "success")
            except sqlite3.IntegrityError:
                flash("Kullanıcı mevcut", "warning")
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u, p = request.form.get('username'), request.form.get('password')
        if not u or not p:
            flash("Boş alan bırakmayın", "danger"); return redirect('/login')
        a = check_user(u, p)
        if a:
            session.update({'logged_in': True,
                            'is_admin': a['role'] == 'admin',
                            'uid': a['id'],
                            'username': u})
            with get_db() as c:
                c.execute("INSERT INTO logins(user_id,ts) VALUES(?,?)",
                          (a['id'], datetime.utcnow().isoformat()))
            return redirect(url_for('admin' if a['role'] == 'admin' else 'home'))
        flash("Hatalı bilgi", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear(); return redirect('/login')

@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        t = request.form.get('message', '').strip()
        if not t:
            flash("Lütfen semptomlarınızı yazarak gönderin.", "warning")
        else:
            session['responses'] = extract_symptoms(t)
            return redirect(url_for('predict'))
    return render_template('index.html', finished=False)

@app.route('/predict')
@login_required
def predict():
    c = session['responses']
    logging.info("Model‣Girdi : %s", c)

    df = pd.DataFrame([{
        'Age': c['age'], 'Gender': c['gender'],
        'Blood Pressure': c['blood_pressure'],
        'Cholesterol Level': c['cholesterol'],
        'Smoking': c['smoking'], 'Alcohol': c['alcohol'], 'Stress': c['stress'],
        'Fever': c['fever'], 'Cough': c['cough'],
        'Fatigue': c['fatigue'], 'Difficulty Breathing': c['difficulty_breathing']}])

    proba     = model.predict_proba(df)[0]
    idx       = proba.argsort()[-10:][::-1]
    diseases  = label_encoder.inverse_transform(idx)
    probs     = proba[idx]
    result    = list(zip(diseases, probs))
    logging.info("Model‣Sonuç : %s", result)

    recs = generate_recommendations(diseases)
    ts   = datetime.utcnow().isoformat()
    with get_db() as db:
        for d, p in result:
            db.execute("""INSERT INTO predictions(user_id,disease,prob,ts)
                          VALUES(?,?,?,?)""",
                       (session['uid'], d, float(p), ts))

    session['recommendations_for_user'] = recs
    return render_template('result.html',
                           prediction=result,
                           recommendations=recs)

@app.route('/recommendations')
@login_required
def rec_page():
    flash("Kesin tanı için sağlık kuruluşuna başvurun.", "warning")
    return render_template('recommendations.html',
                           recommendations=session.get('recommendations_for_user', {}))

# ---------------- Admin Panel  ----------------------------------------
@app.route('/admin')
@login_required
@admin_required
def admin():
    today = datetime.utcnow().date().isoformat()

    with get_db() as c:
        users_cnt = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        preds_cnt = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        today_cnt = c.execute(
            "SELECT COUNT(*) FROM predictions WHERE ts LIKE ?||'%'", (today,)
        ).fetchone()[0]

        top = c.execute("""
            SELECT disease, COUNT(*) cnt
            FROM predictions
            GROUP BY disease
            ORDER BY cnt DESC
            LIMIT 10
        """).fetchall()

        user_stats = c.execute("""
            SELECT u.username,
                   COUNT(l.id),
                   COALESCE(MAX(l.ts), '-')
            FROM users u
            LEFT JOIN logins l ON l.user_id = u.id
            GROUP BY u.id
            ORDER BY 2 DESC
        """).fetchall()

        predictions = c.execute("""
            SELECT p.id,
                   u.username,
                   p.disease,
                   p.prob,
                   p.ts
            FROM predictions p
            JOIN users u ON u.id = p.user_id
            ORDER BY p.ts DESC
            LIMIT 100
        """).fetchall()

        # son 30 gün
        days, loginSeries, predSeries = [], [], []
        for i in range(29, -1, -1):
            day_iso = (datetime.utcnow() - timedelta(days=i)).date().isoformat()
            days.append(day_iso[5:])
            loginSeries.append(
                c.execute("SELECT COUNT(*) FROM logins WHERE ts LIKE ?||'%'", (day_iso,)).fetchone()[0])
            predSeries.append(
                c.execute("SELECT COUNT(*) FROM predictions WHERE ts LIKE ?||'%'", (day_iso,)).fetchone()[0])

    counts = {'users': users_cnt,
              'preds': preds_cnt,
              'today': today_cnt}

    charts = {
        'topLabels':   [r[0] for r in top] or ['Veri yok'],
        'topCounts':   [r[1] for r in top] or [0],
        'days':        days,
        'loginSeries': loginSeries,
        'predSeries':  predSeries
    }

    return render_template('admin.html',
                           counts=counts,
                           charts=json.dumps(charts),
                           user_stats=user_stats,
                           predictions=predictions)

# ------ satır silen AJAX endpoint’i -----------------------------------
@app.route('/delete_preds', methods=['POST'])
@login_required
@admin_required
def delete_preds():
    ids = request.json.get('ids', [])
    if not ids:
        return jsonify({'status': 'no_ids'}), 400
    with get_db() as c:
        q = f"DELETE FROM predictions WHERE id IN ({','.join(['?']*len(ids))})"
        c.execute(q, ids)
    return jsonify({'status': 'ok', 'deleted': len(ids)})

# -------------------------- Main ---------------------------------------
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
