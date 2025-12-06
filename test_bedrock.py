import boto3, json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [{"role": "user", "content": "Say hi in 5 words."}],
    "max_tokens": 50
}

resp = client.invoke_model(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    body=json.dumps(body),
    contentType="application/json",
    accept="application/json"
)

print(json.loads(resp["body"].read()))
