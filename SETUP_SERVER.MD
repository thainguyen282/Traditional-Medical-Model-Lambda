install ngrok 
cd server

uvicorn main:app --host 0.0.0.0 --port 8000
set up ngrok https://dashboard.ngrok.com/get-started/setup/linux

ngrok http http://0.0.0.0:8000

