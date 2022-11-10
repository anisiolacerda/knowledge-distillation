pip list --format=freeze > requirements.txt
pip-sync requirements.txt --pip-args "--no-cache-dir --no-deps"pip-sync requirements.txt --pip-args "--no-cache-dir --no-deps"
