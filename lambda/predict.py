import json
import typing as t


from common.googlify import googlify


def lambda_handler(event: dict, context: t.Any) -> dict[str, t.Any]:
    data: dict[str, t.Any] = json.loads(event["body"])
    return googlify(data)
