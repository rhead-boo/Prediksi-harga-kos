
import os
import json
import joblib
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt


# ==========================================
# KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_PATH = BASE_DIR / "model_kos.joblib"
DATASET_PATH = BASE_DIR / "kos_madiun.csv"

def home(request):
    return render(request, "prediksi/index.html")


@require_GET
def api_predict(request):
    try:
        jarak = float(request.GET.get("jarak"))
        rating = float(request.GET.get("rating"))
        ac = request.GET.get("ac", "").lower()

        if not 0.6 <= jarak <= 5.0:
            return JsonResponse(
                {"error": "Jarak harus antara 0.6 sampai 5 km"},
                status=400
            )

        if not 3.1 <= rating <= 4.9:
            return JsonResponse(
                {"error": "Rating harus antara 3.1 sampai 4.9"},
                status=400
            )

        if ac not in ["ya", "tidak"]:
            return JsonResponse(
                {"error": "Pilihan AC tidak valid"},
                status=400
            )

        ac_num = 1 if ac == "ya" else 0

        model = joblib.load(MODEL_PATH)

        hasil = model.predict([[jarak, rating, ac_num]])[0]

        return JsonResponse({
            "harga_ribu": round(float(hasil), 2),
            "harga_rupiah": round(float(hasil) * 1000),
            "pesan": "Prediksi berhasil"
        })

    except (ValueError, TypeError):
        return JsonResponse(
            {"error": "Input prediksi tidak valid"},
            status=400
        )

    except FileNotFoundError:
        return JsonResponse(
            {"error": "File model_kos.joblib tidak ditemukan"},
            status=500
        )

    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=500
        )


# API DATA GRAFIK

@require_GET
def api_chart_insight(request):
    try:
        df = pd.read_csv(DATASET_PATH)

        rata_ac = df[
            df["ac"].str.lower() == "ya"
        ]["harga"].mean()

        rata_non_ac = df[
            df["ac"].str.lower() == "tidak"
        ]["harga"].mean()

        if rata_ac > rata_non_ac:
            penjelasan = (
                "Berdasarkan dataset, kos dengan fasilitas AC "
                "memiliki rata-rata harga lebih tinggi."
            )

        elif rata_ac < rata_non_ac:
            penjelasan = (
                "Berdasarkan dataset, kos tanpa fasilitas AC "
                "memiliki rata-rata harga lebih tinggi."
            )

        else:
            penjelasan = (
                "Rata-rata harga kedua kelompok kos sama."
            )

        return JsonResponse({
            "labels": ["Dengan AC", "Tanpa AC"],
            "data": [
                round(float(rata_ac), 2),
                round(float(rata_non_ac), 2)
            ],
            "satuan": "ribu rupiah per bulan",
            "penjelasan": penjelasan
        })

    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=500
        )


# API AI GROQ - ANALISIS HASIL PREDIKSI

@csrf_exempt
@require_POST
def api_predict_insight(request):
    try:
        if not GROQ_API_KEY:
            return JsonResponse(
                {"error": "API Key Groq belum tersedia"},
                status=500
            )

        body = json.loads(request.body)
        prediksi = body.get("prediksi", {})

        client = Groq(api_key=GROQ_API_KEY)

        hasil = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah analis harga kos di Madiun. "
                        "Jelaskan hasil prediksi Machine Learning "
                        "dalam bahasa Indonesia yang mudah dipahami. "
                        "Bahas nominal prediksi per bulan, jarak kos, "
                        "rating, dan fasilitas AC yang dipilih. "
                        "Jelaskan arti hasil prediksi bagi pengguna. "
                        "Jangan mengarang pengaruh variabel atau "
                        "menyatakan hubungan sebab-akibat. "
                        "Tegaskan bahwa harga merupakan estimasi, "
                        "bukan harga pasti. "
                        "Berikan penjelasan yang informatif dalam "
                        "beberapa kalimat. Gunakan paragraf biasa "
                        "tanpa Markdown."
                    )
                },
                {
                    "role": "user",
                    "content": f"Data prediksi pengguna: {prediksi}"
                }
            ],
            temperature=0.5,
            max_completion_tokens=500
        )

        return JsonResponse({
            "penjelasan": hasil.choices[0].message.content
        })

    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=500
        )


# API AI GROQ - ANALISIS GRAFIK

@csrf_exempt
@require_POST
def api_chart_ai(request):
    try:
        if not GROQ_API_KEY:
            return JsonResponse(
                {"error": "API Key Groq belum tersedia"},
                status=500
            )

        body = json.loads(request.body)
        grafik = body.get("grafik", {})

        client = Groq(api_key=GROQ_API_KEY)

        hasil = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah analis data harga kos Madiun. "
                        "Jelaskan grafik perbandingan harga kos "
                        "dengan AC dan tanpa AC dalam bahasa Indonesia. "
                        "Sebutkan nilai rata-rata kedua kelompok, "
                        "jelaskan perbedaannya, dan berikan kesimpulan "
                        "berdasarkan data. "
                        "Jangan mengarang informasi atau menyatakan "
                        "hubungan sebab-akibat. "
                        "Berikan penjelasan yang informatif dalam "
                        "beberapa kalimat. Gunakan paragraf biasa "
                        "tanpa Markdown."
                    )
                },
                {
                    "role": "user",
                    "content": f"Data grafik: {grafik}"
                }
            ],
            temperature=0.5,
            max_completion_tokens=500
        )

        return JsonResponse({
            "penjelasan": hasil.choices[0].message.content
        })

    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=500
        )
