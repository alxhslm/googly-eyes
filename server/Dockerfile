FROM python:3.11-slim

RUN pip install poetry==1.6.0

WORKDIR /app

# Copy the poetry files into our working directory
COPY server/pyproject.toml server/poetry.lock ./
RUN poetry config virtualenvs.in-project false
RUN poetry install

# Copy any python files and the model we had to the working directory of Docker
COPY server/app.py ./
COPY common ./common
COPY retinaface/__init__.py ./retinaface/__init__.py
COPY retinaface/commons ./retinaface/commons
COPY retinaface/detect.py ./retinaface/detect.py
COPY retinaface/retinaface.tflite ./retinaface/retinaface.tflite

# We need to expose the 8000 port because we're not able to communicate with Docker outside it
EXPOSE 8000

# Launch the server
CMD ["poetry", "run", "waitress-serve", "--host=0.0.0.0" "port=8000", "app:app"]
