import os

# Never send traces from test runs to a real Langfuse project, even if .env has keys.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
