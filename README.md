# 🎬 Movie Recommendation System

A web-based **Movie Recommendation System** built with **Python, Flask, and the MovieLens dataset**.
It suggests movies to users using **collaborative filtering (SVD)** and provides a **genre-based fallback** for new users.

---

## 📌 Project Overview

The Movie Recommendation System helps users discover movies they are likely to enjoy.
It is designed to handle both returning users (who have rated movies before) and new users (first-time visitors).

* Returning users receive recommendations based on **collaborative filtering**, where the system learns from similar users’ preferences.
* New users (who have not rated any movies yet) can still receive recommendations by selecting their preferred **genres**, solving the **cold-start problem**.

The project combines **machine learning** with a simple **Flask web app** and provides an intuitive HTML + CSS frontend.

---

## 🔑 Features

* ✅ Personalized recommendations with **SVD collaborative filtering**
* ✅ **Cold-start solution** using genre preferences
* ✅ **Flask-based web interface** with forms and results
* ✅ Clean, responsive **HTML + CSS UI**

---

## 🛠 Tech Stack

* **Backend:** Python, Flask, Pandas, Scikit-learn, Surprise
* **Frontend:** HTML, CSS
* **Dataset:** MovieLens (ratings + movie metadata)

---

## 🚀 How It Works

1. **Data Preprocessing** → Load MovieLens dataset, clean ratings, map user & movie IDs.
2. **Model Training** → Train **SVD (Singular Value Decomposition)** model for collaborative filtering.
3. **Recommendation Logic**:

   * If user has rated → personalized predictions.
   * If user is new → fallback to genre-based filtering.
4. **Web Interface** → Flask app with a simple HTML form to input User ID or preferred genres.

---

## 📂 Project Structure

```
Movie_Recommendation_System/
│── movie_pipeline.py        # Train + evaluate recommendation model
│── app.py                   # Flask web app
│── templates/
│   └── index.html           # HTML frontend
│── static/
│   └── style.css            # CSS for styling
│── movie_outputs/           # Saved model outputs
│── README.md                # Documentation
```

---

## 🖥️ Running the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Movie_Recommendation_System.git
   cd Movie_Recommendation_System
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:

   ```bash
   python movie_pipeline.py
   ```

4. Run the Flask app:

   ```bash
   python app.py
   ```

5. Open in browser:

   ```
   http://127.0.0.1:5000/
   ```

---

## 📈 Future Improvements

* 🔹 Add **hybrid recommendation** (content + collaborative filtering).
* 🔹 Deploy on **Heroku / Render / AWS** for public access.
* 🔹 Add **user authentication & profile saving**.

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

⚡ Ready to use. Just clone, install, and run.

---

~ Jinit Yadav
