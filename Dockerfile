# use lightweight python
FROM python:3.9-slim

# set working dir
WORKDIR /app

# copy code
COPY . .

# install system dependencies
# RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# expose port
EXPOSE 9003

# run fastapi server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9003"]
