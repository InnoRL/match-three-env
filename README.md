# Match Three Gymnax Environment

## Getting Started

Perform an editable install of the environment:
```bash
cd match3_env
pip install -e .
```

## Demo

Run demo:
```bash
cd match3_demo
pip install -e .

uvicorn --host 0.0.0.0 --port 8001 --reload src.match3_demo.main:app
```
