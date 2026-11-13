# Section 7 — Streamlit (Tab 1 only)

This is the first working stage of the five-tab app. It implements the delivery overview and duration estimate. Remaining tabs have not been implemented.

Run locally from this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use a trusted saved model only. `scikit-learn==1.6.1` matches the saved model.

The full `images.npz` training arrays were intentionally excluded from this app package. The project data are generated teaching scenarios, not operational deliveries.
