response = bedrock_runtime.invoke_model(...)

usage = response["usage"]
input_tokens = usage.get("input_tokens", 0)
output_tokens = usage.get("output_tokens", 0)

# Example cost rate – adjust per your model:
cost_per_1k_input = 0.003
cost_per_1k_output = 0.015

estimated_cost = (
    (input_tokens / 1000) * cost_per_1k_input +
    (output_tokens / 1000) * cost_per_1k_output
)

return {
    "text": answer,
    "usage": {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimated_cost
    }
}
