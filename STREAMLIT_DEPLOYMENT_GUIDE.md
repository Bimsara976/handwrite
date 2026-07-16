# Streamlit Community Cloud Deployment

1. Create a public GitHub repository such as `com763-digit-recognition`.
2. Upload every project file and folder to its repository root. Confirm that the notebook, `app.py`, `requirements.txt`, `.joblib`, JSON file, and `artifacts` folder are visible.
3. Test locally with `pip install -r requirements.txt` followed by `streamlit run app.py`.
4. Sign in at https://share.streamlit.io using GitHub.
5. Choose **Create app**, then select the repository and branch.
6. Set the main file path to `app.py`, select an app URL, and deploy. No secrets are required.
7. Open the HTTPS URL, upload a clear single-digit image, and confirm that the prediction and probability chart appear.
8. Put the deployed URL in the technical report you write yourself.

## Updating and troubleshooting

- Push later changes to the same branch; the cloud service normally rebuilds automatically.
- For `ModuleNotFoundError`, add the missing package to `requirements.txt`, push, and reboot.
- For a missing model error, confirm `digit_recognition_pipeline.joblib` is in the root with matching capitalisation.
- Retain the tested dependency versions supplied here to minimise build conflicts.
- Poor phone-image predictions can reflect domain shift; use one thick, centred dark digit on a plain light background.
