FROM python:3.11-slim

RUN pip install poetry==1.6.0

WORKDIR /app

# Copy the poetry files into our working directory
COPY dashboard/pyproject.toml dashboard/poetry.lock ./
RUN poetry config virtualenvs.in-project false
RUN poetry install

# Copy any python files and the model we had to the working directory of Docker
COPY dashboard/app.py ./
COPY common ./common

# We need to expose the 8501 port because we're not able to communicate with Docker outside it
EXPOSE 8501

# Set the URL for the server
ARG SERVER_URL
ENV SERVER_URL=$SERVER_URL

# Launch the dashboard
CMD ["poetry", "run" , "streamlit", "run", "app.py", "--server.port", "8501"]
