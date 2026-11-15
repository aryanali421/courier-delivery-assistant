# Parcel / Studio — Project 05

A five-tab Streamlit demonstration for the **Courier Delivery Planning and Feedback Assistant** assignment. Uses the saved Decision Tree, KMeans and CNN models trained on the supplied generated classroom dataset; the provided English comments; and the included offline NLTK VADER lexicon. **No training happens in the app.**

## Tabs
1. Delivery overview and duration estimate — input validation and saved model predictions.
2. Delivery profiles — three KMeans groups and supplied group chart.
3. Vehicle image — sample/uploaded image, RGB 32×32 CNN classification and manual correction.
4. Customer comments — VADER sentiment, supplied keyword topics, and example comment dropdown.
5. Service summary — default offline template, optionally free Groq summary.

## Run locally

Use a Python version supported by the dependency versions (the models were trained with Python 3.13, scikit-learn 1.6.1, and TensorFlow 2.21). From this directory:

```bash
python -m venv .venv
# Activate your environment, then:
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a free GitHub repository and upload this folder (keep `app.py` at the repository root).
2. Do **not** upload your RAR/ZIP source archives or the full `images.npz` training arrays.
3. Create a new app at Streamlit Community Cloud, select the repository and `app.py`, and use a Python runtime compatible with the dependencies.
4. Optional: in Streamlit Cloud **Settings → Secrets** add:

```toml
GROQ_API_KEY = "your-free-plan-key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

No Groq key is needed for the default offline app. Keep keys private and do not commit `.streamlit/secrets.toml`. The live model ID and free quota may change. A missing key, network error, quota error, or empty response leaves the offline summary usable.

## Test checklist
- [ ] Run without Groq secrets; all five tabs remain available.
- [ ] Select a supplied example, get a prediction in minutes and a cluster group.
- [ ] Enter zero for distance; see a validation error (valid range: 1–30 km).
- [ ] Check a supplied car/truck image; confirm or correct the prediction.
- [ ] Submit an empty manual comment; see `empty`/excluded response.
- [ ] Check supplied comment topics and matched keywords.
- [ ] Turn off network / simulate Groq error; offline summary remains usable.
- [ ] Restart the app; saved models load without retraining.
- [ ] Test on a second computer with this README.

## Scope and limitations
- All tabular values are *generated classroom scenarios*, not operational courier records.
- The image classifier is trained on low-resolution CIFAR-10 car/truck images; confidence-like output is not a calibrated probability and must be manually verified.
- Comments are authored English examples rather than a customer survey. VADER does not fully understand sarcasm, Urdu, or Roman Urdu.
- Clustering groups are not rankings, and estimated times are not guarantees.
- No live tracking, route optimization, maps, driver ranking, automated messages, or database.
- The prompt-comparison report from the training notebook still requires completed manual evidence; this app does not invent those results.

See `source_notices/` for licenses and attribution and `reports/` for measured training outputs.
