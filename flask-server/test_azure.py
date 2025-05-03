
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = [
    'OPENAI_API_KEY',
    'AZURE_ENDPOINT',
    'AZURE_CONN_STRING',
    'AZURE_STORAGE_KEY',
    'AZURE_COGNITIVE_SEARCH_ENDPOINT',
    'AZURE_COGNITIVE_SEARCH_API_KEY',
    'MONGO_URI'
]

missing_vars = []
for var in required_vars:
    value = os.environ.get(var)
    if not value:
        missing_vars.append(var)
    else:
        print(f"{var}: {'*' * 4}{value[-4:] if len(value) > 4 else ''}")

if missing_vars:
    print(f"Missing environment variables: {', '.join(missing_vars)}")
    print("Please create a .env file with these variables")
else:
    print("All required environment variables are set!")
