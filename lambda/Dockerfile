FROM public.ecr.aws/lambda/python:3.11

RUN pip install poetry==1.6.0

# Copy the poetry files into our working directory
COPY server/pyproject.toml server/poetry.lock ./
RUN poetry config virtualenvs.in-project false
RUN poetry install

# Copy any python files and the model we had to the working directory of Docker
COPY lambda/predict.py ./
COPY common ./common
COPY retinaface/retinaface.tflite ./

CMD [ "predict.lambda_handler" ]
