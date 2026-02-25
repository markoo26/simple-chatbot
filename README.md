
# 💡 Before you start


## 🌲 UV venv setup

```
git clone https://github.com/markoo26/simple-chatbot.git
pip install uv
uv venv
.venv\Scripts\activate
pip install -e .
```

## 🚓 Create a dynamic database with dummy parking data called `parking_system.db`
```
python scripts/create_parking_db.py
```

## ➡️ Create a vector database processing three parking-related documents
```
python scripts/create_vector_db.py
```
