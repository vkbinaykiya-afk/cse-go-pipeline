#!/bin/bash
# Local Mac launcher — secrets injected via launchd plist EnvironmentVariables, not here
export HOME=/Users/vaibhavbinaykiya
export PYTHONNOUSERSITE=1
cd /Users/vaibhavbinaykiya/cse-go-pipeline
exec /Users/vaibhavbinaykiya/csego_venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000 --loop asyncio --http h11
