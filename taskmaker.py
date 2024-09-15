from openai import OpenAI
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:3301/v1", api_key="lm-studio")

# Define sectors and difficulties
sectors = ["medical", "science", "math", "coding", "factual", "heuristics"]
difficulties = [
    "normal regularly encountered",
    "very hard and challenging for anyone but the best to solve",
    "nearly impossible",
    "extremely challenging but theoretically possible"
]

def clean_text(text):
    # Replace '\n' with actual newlines, remove extra spaces
    cleaned = text.replace('\\n', '\n').strip()
    # Remove any remaining '\' characters
    cleaned = cleaned.replace('\\', '')
    # Replace multiple newlines with a single newline
    while '\n\n' in cleaned:
        cleaned = cleaned.replace('\n\n', '\n')
    return cleaned

def generate_problem(sector, difficulty):
    prompt = f"""Generate a complex, open-ended problem in the {sector} sector, similar in structure and conciseness to this example:

    "In the book 'Advances in Active Portfolio Management' there is a section on clustering the returns of stocks based on log exponential time-weighted returns into groups using k-means clustering. Given a folder that contains files with the names of {{ticker}}.csv with the csv containing the columns of Date, Open, High, Low, Close, Volume. Can you replicate the k-means clustering of log returns using the data in the 'Stock_Data' folder? Ideally, you should cluster the stocks into less than 10 groups and create a new csv that has the ticker of the stock and the cluster it belongs to called 'clustered_stocks.csv'"

    The problem should be {difficulty} and should have a realistic solution. 
    Include names of file paths like the example above.
    For coding problems, specify the language and any relevant libraries or frameworks.
    Keep the problem statement concise but detailed enough to be actionable.
    Do not use '\\n' for line breaks; use proper formatting instead."""

    response = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        messages=[
            {"role": "system", "content": "You are an AI that generates focused, complex, open-ended problems for various sectors."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    
    return clean_text(response.choices[0].message.content.strip())

# Generate problems
data = []
total_problems = 100  # Set this to 500 for the full dataset
for i in tqdm(range(total_problems)):
    sector = sectors[i % len(sectors)]
    difficulty = difficulties[i % len(difficulties)]
    problem = generate_problem(sector, difficulty)
    data.append({
        'index': i,
        'sector': sector,
        'difficulty': difficulty,
        'hardproblemprompt': problem
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to Parquet file
table = pa.Table.from_pandas(df)
pq.write_table(table, 'complextasks.parquet')

# Save to CSV file
df.to_csv('complextasks.csv', index=False)

print(f"Files 'complextasks.parquet' and 'complextasks.csv' have been created with {total_problems} complex problems.")

# Display a few examples
print("\nSample problems:")
for _, row in df.sample(min(3, len(df))).iterrows():
    print(f"\nSector: {row['sector']}")
    print(f"Difficulty: {row['difficulty']}")
    print(f"Problem:\n{row['hardproblemprompt']}")
    print("-" * 80)

# Optional: Basic statistics
print("\nProblem distribution:")
print(df['sector'].value_counts())
print("\nDifficulty distribution:")
print(df['difficulty'].value_counts())


