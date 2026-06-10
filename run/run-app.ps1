# python3 -m uvicorn app:app --reload
# uvicorn --host=localhost --app-dir=. app:app --port=8000 --reload
uvicorn --host=localhost --app-dir=. app:app --port=8000
