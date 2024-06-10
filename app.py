import os
from dashboard.app import run_dashboard

from aws_requests_auth.aws_auth import AWSRequestsAuth

resource = "dzdzxpkrlmjj74daububqcowty0fbiev"
region = "eu-west-2"
host = f"{resource}.lambda-url.{region}.on.aws"
url = f"https://{host}/"

auth = AWSRequestsAuth(
    aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    aws_host=host,
    aws_region=region,
    aws_service="lambda",
)


run_dashboard(url, auth)
