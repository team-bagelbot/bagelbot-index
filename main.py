import os
import openai
import asyncio
from google.cloud import storage
from llama_index import GPTSimpleVectorIndex, SimpleWebPageReader
from time import perf_counter
import requests

# Set the OPENAI_API_KEY environment var in CloudRun job via Secrets reference

PROJECT_ID = os.environ['PROJECT_ID']
MODEL = 'text-davinci-003'
INDEX_VECTOR_FILE = 'index_vector.json'
QUESTIONS_FILE = 'questions.txt'
HASH_FILE = 'sha256'
INDEX_DATA_URL = os.environ['INDEX_DATA_URL'] # Set the INDEX_DATA_URL environment var in CloudRun job
print(f'{PROJECT_ID}: INDEX_DATA_URL: {INDEX_DATA_URL}')


async def upload_file(bucket_name, source_file_name, destination_blob_name) -> None:

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    try:
        blob.upload_from_filename(source_file_name)
    except Exception as e:
        print(f'{PROJECT_ID}: error uploading {source_file_name} to {destination_blob_name} in bucket {bucket_name}: {e}')
        raise e
    else:
        print(
            f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}."
        )

async def main() -> None:
    print(f'{PROJECT_ID}: starting index build')

    # request url
    try:
        urls = requests.get(INDEX_DATA_URL).text.splitlines()
    except Exception as e:
        print(f'{PROJECT_ID}: error requesting {INDEX_DATA_URL}: {e}')
        raise e
    else:
        documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
        print(f'{PROJECT_ID}: loaded web pages via SimpleWebPageReader')

    # build the index
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk(INDEX_VECTOR_FILE)
    print(f'{PROJECT_ID}: saved index to: {INDEX_VECTOR_FILE}')

    # hash the index
    import hashlib
    hashed = hashlib.sha256(str(index).encode()).hexdigest()

    #save the hash to disk
    try:
        with open(HASH_FILE, 'w') as f:
            f.write(str(hashed).strip())
    except Exception as e:
        print(f'{PROJECT_ID}: error writing {HASH_FILE} to disk: {e}')
        raise e

    # summarize the index
    index_summary = index.query(
        'What is a summary of this collection of text?',
        response_mode="tree_summarize")
    print(f'{PROJECT_ID}: created index summary')

    # build questions
    prompt = f"What are 3 questions that can be answered only using the following summary? {str(index_summary)}"
    response = openai.Completion.create(
        prompt=prompt, model=MODEL, temperature=0.0, max_tokens=128
    )
    questions = response.choices[0].text

    # create list from questions using \n as delimiter
    questions = questions.split('\n')

    # remove any empty items from the list
    questions = [x for x in questions if x]

    # append questions to summary with each question on a new line
    index_questions = f'{questions[0]}\n{questions[1]}\n{questions[2]}'

    # save the summary to disk
    try:
        with open(QUESTIONS_FILE, 'w') as f:
            f.write(str(index_questions).strip())
    except Exception as e:
        print(f'{PROJECT_ID}: error writing {QUESTIONS_FILE} to disk: {e}')
        raise e

    time_before_async_upload = perf_counter()
    result = await asyncio.gather(
        upload_file(PROJECT_ID, INDEX_VECTOR_FILE, INDEX_VECTOR_FILE),
        upload_file(PROJECT_ID, HASH_FILE, HASH_FILE),
        upload_file(PROJECT_ID, QUESTIONS_FILE, QUESTIONS_FILE))
    print(f'{PROJECT_ID}: async upload time: {perf_counter() - time_before_async_upload}')
    print(result)
    print(f'{PROJECT_ID}: finished index build')

if __name__ == '__main__':
    asyncio.run(main())


