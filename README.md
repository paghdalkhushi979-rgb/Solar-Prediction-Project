# Smart Attendance System — Face Recognition (Streamlit + Deep Learning)

A college-project-ready attendance system: register students with a webcam
photo, then mark attendance automatically by recognizing their face —
no manual entry needed.

## How it works (architecture)

```
Webcam photo (st.camera_input)
        │
        ▼
DeepFace (Facenet deep learning model)
        │  -> converts face into a numeric vector ("embedding")
        ▼
numpy cosine similarity
        │  -> compares the new face's embedding to every
        │     stored student embedding
        ▼
pandas
        │  -> logs match as attendance row in attendance.csv
        ▼
Streamlit (with custom CSS) -> dashboard, charts, records table
```

- **Deep learning**: `DeepFace` runs a pretrained CNN (Facenet) to turn a
  face photo into a 128-dimension vector. Two photos of the same person
  produce vectors that are close together; different people produce
  vectors that are far apart.
- **numpy**: used for the cosine-similarity comparison and for the pandas
  data underneath.
- **pandas**: stores the student list (`students.csv`) and the attendance
  log (`attendance.csv`), and powers the records/download screen.
- **matplotlib**: dashboard charts (attendance % per student, today's
  present/absent split).
- **Streamlit + custom CSS** (`style.css`): the UI — no separate HTML/JS
  frontend needed, `st.markdown(..., unsafe_allow_html=True)` injects the
  CSS to theme it.

## Setup

1. Install Python 3.9–3.11 (TensorFlow/DeepFace don't yet support the very
   latest Python versions — 3.10 is a safe bet).
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   First install will take a few minutes — TensorFlow is a large package,
   and DeepFace downloads the Facenet model weights (~90 MB) the first
   time you register a student.
4. Run the app:
   ```bash
   streamlit run app.py
   ```
   It opens in your browser. Your laptop's webcam is used through the
   browser's camera permission — allow it when asked.

## Using it

1. **➕ Register Student** — enter roll number, name, class, capture a
   clear front-facing photo, click Register. Do this once per student.
2. **📷 Mark Attendance** — student stands in front of the camera, clicks
   capture. If recognized, attendance is logged instantly with the
   current date/time. Marking again the same day won't duplicate.
3. **🏠 Dashboard** — live counts and charts.
4. **📋 Attendance Records** — filter by date, download CSV for your
   report/submission.

## Tuning accuracy

In `utils.py`:
```python
SIMILARITY_THRESHOLD = 0.45
```
- Real students getting rejected too often → lower this (e.g. `0.35`).
- Different people getting matched to each other → raise this (e.g. `0.6`).
- Test with a few students and adjust based on what you see — this is a
  normal step to mention in your project report.

## Project report talking points

- Face detection + embedding extraction: **DeepFace (Facenet CNN)**,
  a deep learning model pretrained for face recognition.
- Similarity search: cosine similarity via **numpy**.
- Data layer: **pandas** DataFrames persisted as CSV (`students.csv`,
  `attendance.csv`), embeddings persisted via `pickle`.
- Frontend/UI: **Streamlit**, styled with custom CSS.
- Visualization: **matplotlib** bar chart (attendance %) and pie chart
  (today's present/absent).

## Folder structure

```
attendance_system/
├── app.py                # Streamlit UI (all 4 pages)
├── utils.py               # face recognition + data logic
├── style.css               # custom UI theming
├── requirements.txt
├── README.md
└── data/                   # created automatically on first run
    ├── student_photos/
    ├── students.csv
    ├── attendance.csv
    └── embeddings.pkl
```
