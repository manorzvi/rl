import os
from google.cloud import storage


def create_bucket(bucket_name):
    """Creates a new bucket."""

    storage_client = storage.Client()

    bucket = storage_client.create_bucket(bucket_name)

    print("Bucket {} created".format(bucket.name))


def explicit_create_bucket(bucket_name : str, service_account_key_file : str):
    """Creates a new bucket."""

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)
    
    bucket = storage_client.create_bucket(bucket_name)

    print("Bucket {} created".format(bucket.name))


def list_buckets():
    """List current project storage buckets"""
    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())
    return buckets


def explicit_list_buckets(service_account_key_file : str):

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)

def check_google_app_cred():
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        return True
    else:
        return False



