from kfp import dsl,compiler
from google.cloud import aiplatform
from google.cloud import storage
from dotenv import load_dotenv
import os
from pathlib import Path


load_dotenv()

cred_filename = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_FILENAME")
if cred_filename:
    print(f'RUNNING LOCAL. ADAPTING LOADING PROCESS')
    project_root = Path(__file__).parent
    os.chdir(project_root)
    dotenv_path = project_root.parent / '.env'
    load_dotenv(dotenv_path=dotenv_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root.parent / cred_filename)

# ---- SETUP ----
PROJECT_ID = 'sibr-market'
REGION = 'europe-west1'
BUCKET_NAME = 'sibr-market'
BUCKET_URI = f'gs://{BUCKET_NAME}'
REPO = 'sibr-market-repo'

SCRAPING_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/scraping:latest"
GEOCODING_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/geocoding:latest"
MODELING_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/modeling:latest"

# ---- PIPELINE COMPONENTS ----
@dsl.container_component
def run_scraping():
    return dsl.ContainerSpec(
        image=SCRAPING_IMAGE_URI,
        command=['python', 'main.py'],
    )

@dsl.container_component
def run_geocoding(): # Endret
    return dsl.ContainerSpec(
        image=GEOCODING_IMAGE_URI,
        command=['python', 'main.py'],
    )

@dsl.container_component
def run_predict(): # Endret
    return dsl.ContainerSpec(
        image=MODELING_IMAGE_URI,
        command=['python', 'main.py'],
        args=['--run-predict']
    )

@dsl.container_component
def run_clean(): # Endret
    return dsl.ContainerSpec(
        image=MODELING_IMAGE_URI,
        command=['python', 'main.py'],
        args=['--run-clean']
    )

# ---- PIPELINE DEFINITION ----
@dsl.pipeline(
    name='sibr-market-pipeline',
    description='Pipeline for scraping, geocoding, and cleaning/predicting data for SIBR Market',
    pipeline_root=BUCKET_URI
)
def create_pipeline(
):
    # Step 1: Run scraping
    # Send image-parameteren inn i komponenten
    scraping_task = run_scraping()
    scraping_task.set_display_name('1. Scraping Data')
    scraping_task.set_caching_options(False)

    # Step 2: Run cleaning
    cleaning_task = run_clean().after(scraping_task)
    cleaning_task.set_display_name('2. Cleaning Data')

    # Step 2: Run geocoding
    geocoding_task = run_geocoding().after(cleaning_task)
    geocoding_task.set_display_name('3. Geocoding Addresses')
    geocoding_task.set_caching_options(False)

    # Step 3: Run cleaning and prediction
    clean_predict_task = run_predict().after(geocoding_task)
    clean_predict_task.set_display_name('4. Predict Data')
    clean_predict_task.set_caching_options(False)

if __name__ == '__main__':

    PIPELINE_JSON = 'sibr_market_pipeline.json'
    compiler.Compiler().compile(
        pipeline_func=create_pipeline,
        package_path=PIPELINE_JSON
    )
    aiplatform.init(project=PROJECT_ID,
                    location=REGION,)
    job = aiplatform.PipelineJob(
        display_name='sibr-market-pipeline',
        template_path=PIPELINE_JSON,
        pipeline_root=BUCKET_URI,
    )

    client = storage.Client()
    client.get_bucket(BUCKET_NAME).blob(PIPELINE_JSON).upload_from_filename(PIPELINE_JSON)

    #job.run()
    #kjør denne: gsutil cp sibr_market_pipeline.json gs://sibr-market/