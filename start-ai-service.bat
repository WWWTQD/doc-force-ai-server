@echo off
echo Starting PBL4 AI Service (PP-StructureV3)...
if exist .venv311\Scripts\python.exe (
    .\.venv311\Scripts\python.exe run.py
) else (
    python run.py
)
pause
