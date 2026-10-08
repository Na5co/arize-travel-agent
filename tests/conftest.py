import os

# The tests never call OpenAI, but building the agent needs a key to be set.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
