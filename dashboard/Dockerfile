FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.11

RUN pipx install poetry==1.6.0

WORKDIR /app

# Copy the Pip files into our working directory
COPY ["pyproject.toml", "poetry.lock", "./"]

RUN poetry install

# Copy any python files and the model we had to the working directory of Docker
COPY ["app.py", "./"]

# We need to expose the 8502 port because we're not able to communicate with Docker outside it
EXPOSE 8501

# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["poetry", "run" , "streamlit", "app.py", "--server.port", "8501"]
