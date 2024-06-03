FROM python:3.11-slim

RUN pip install poetry==1.6.0

WORKDIR /app

# Copy the poetry files into our working directory
COPY lambda/pyproject.toml lambda/poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install

# Copy any python files and the model we had to the working directory of Docker
COPY lambda/predict.py ./
COPY common ./common
COPY retinaface/retinaface.tflite ./

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "predict.lambda_handler" ]
