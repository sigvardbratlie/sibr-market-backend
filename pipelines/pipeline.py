from kfp import dsl,compiler
from google.cloud import aiplatform
from google.cloud import storage
from dotenv import load_dotenv
load_dotenv()

# ---- SETUP ----
PROJECT_ID = 'sibr-market'
REGION = 'europe-west1'
BUCKET_NAME = 'sibr-market'
BUCKET_URI = f'gs://{BUCKET_NAME}'
REPO = 'sibr-market-repo'

# ---- PIPELINE COMPONENTS ----
@dsl.container_component
@dsl.container_component
def run_scraping(image: str): # Endret fra ingenting til å ta 'image'
    return dsl.ContainerSpec(
        image=image, # Bruker parameteren her
        command=['python', 'main.py'],
    )

@dsl.container_component
def run_geocoding(image: str): # Endret
    return dsl.ContainerSpec(
        image=image, # Bruker parameteren her
        command=['python', 'main.py'],
    )

@dsl.container_component
def run_clean_predict(image: str): # Endret
    return dsl.ContainerSpec(
        image=image, # Bruker parameteren her
        command=['python', 'main.py'],
        args=['--run_all']
    )

# ---- PIPELINE DEFINITION ----
@dsl.pipeline(
    name='sibr-market-pipeline',
    description='Pipeline for scraping, geocoding, and cleaning/predicting data for SIBR Market',
    pipeline_root=BUCKET_URI
)
def create_pipeline(
    # Definer parametere for pipelinen. Disse kan overstyres ved kjøring.
    scraping_image: str,
    geocoding_image: str,
    mldata_image: str
):
    # Step 1: Run scraping
    # Send image-parameteren inn i komponenten
    scraping_task = run_scraping(image=scraping_image)
    scraping_task.set_display_name('1. Scraping Data')
    scraping_task.set_caching_options(False) # Bra at du har skrudd av caching!

    # Step 2: Run geocoding
    geocoding_task = run_geocoding(image=geocoding_image).after(scraping_task)
    geocoding_task.set_display_name('2. Geocoding Addresses')
    geocoding_task.set_caching_options(False)

    # Step 3: Run cleaning and prediction
    clean_predict_task = run_clean_predict(image=mldata_image).after(geocoding_task)
    clean_predict_task.set_display_name('3. Clean and Predict Data')
    clean_predict_task.set_caching_options(False)

if __name__ == '__main__':


    SCRAPING_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/scraping:latest"
    GEOCODING_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/geocoding:latest"
    MLDATA_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO}/mldata:latest"

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