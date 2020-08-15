import os
from typing import List
import pprint
from google.cloud import storage


def create_bucket(bucket_name : str) -> storage.bucket.Bucket:
    """Creates a new bucket."""

    storage_client = storage.Client()

    bucket = storage_client.create_bucket(bucket_name)

    return bucket


def explicit_create_bucket(bucket_name : str, service_account_key_file : str) -> storage.bucket.Bucket:
    """Creates a new bucket."""

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)

    bucket = storage_client.create_bucket(bucket_name)

    return bucket


def delete_bucket(bucket_name : str):
    """Deletes a bucket. The bucket must be empty."""

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)

    bucket.delete()

    print("Bucket {} deleted".format(bucket.name))


def explicit_delete_bucket(bucket_name: str, service_account_key_file : str):
    """Deletes a bucket. The bucket must be empty."""

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)

    bucket = storage_client.get_bucket(bucket_name)

    bucket.delete()

    print("Bucket {} deleted".format(bucket.name))


def list_buckets() -> List[storage.bucket.Bucket]:
    """List current project storage buckets"""

    storage_client = storage.Client()

    buckets = list(storage_client.list_buckets())

    return buckets


def explicit_list_buckets(service_account_key_file : str) -> List[storage.bucket.Bucket]:
    """List current project storage buckets"""

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())

    return buckets


def bucket_metadata(bucket_name : str):
    """Prints out a bucket's metadata."""

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)

    print("ID: {}".format(bucket.id))
    print("Name: {}".format(bucket.name))
    print("Storage Class: {}".format(bucket.storage_class))
    print("Location: {}".format(bucket.location))
    print("Location Type: {}".format(bucket.location_type))
    print("Cors: {}".format(bucket.cors))
    print("Default Event Based Hold: {}".format(bucket.default_event_based_hold))
    print("Default KMS Key Name: {}".format(bucket.default_kms_key_name))
    print("Metageneration: {}".format(bucket.metageneration))
    print("Retention Effective Time: {}".format(bucket.retention_policy_effective_time))
    print("Retention Period: {}".format(bucket.retention_period))
    print("Retention Policy Locked: {}".format(bucket.retention_policy_locked))
    print("Requester Pays: {}".format(bucket.requester_pays))
    print("Self Link: {}".format(bucket.self_link))
    print("Time Created: {}".format(bucket.time_created))
    print("Versioning Enabled: {}".format(bucket.versioning_enabled))
    print("Labels:")
    pprint.pprint(bucket.labels)


def explicit_bucket_metadata(bucket_name : str, service_account_key_file : str):
    """Prints out a bucket's metadata."""

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)

    bucket = storage_client.get_bucket(bucket_name)

    print("ID: {}".format(bucket.id))
    print("Name: {}".format(bucket.name))
    print("Storage Class: {}".format(bucket.storage_class))
    print("Location: {}".format(bucket.location))
    print("Location Type: {}".format(bucket.location_type))
    print("Cors: {}".format(bucket.cors))
    print("Default Event Based Hold: {}".format(bucket.default_event_based_hold))
    print("Default KMS Key Name: {}".format(bucket.default_kms_key_name))
    print("Metageneration: {}".format(bucket.metageneration))
    print("Retention Effective Time: {}".format(bucket.retention_policy_effective_time))
    print("Retention Period: {}".format(bucket.retention_period))
    print("Retention Policy Locked: {}".format(bucket.retention_policy_locked))
    print("Requester Pays: {}".format(bucket.requester_pays))
    print("Self Link: {}".format(bucket.self_link))
    print("Time Created: {}".format(bucket.time_created))
    print("Versioning Enabled: {}".format(bucket.versioning_enabled))
    print("Labels:")
    pprint.pprint(bucket.labels)


# TODO:  Add return type hint
def list_blobs(bucket_name: str, verbose : bool = False):
    """Lists all the blobs in the bucket."""

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = list(storage_client.list_blobs(bucket_name))

    if verbose:
        for blob in blobs:
            print(blob.name)

    return blobs


# TODO:  Add return type hint
def explicit_list_blobs(bucket_name: str, service_account_key_file : str, verbose : bool = False):
    """Lists all the blobs in the bucket."""

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = list(storage_client.list_blobs(bucket_name))

    if verbose:
        for blob in blobs:
            print(blob.name)

    return blobs


# TODO:  Add return type hint
def list_blobs_with_prefix(bucket_name : str, prefix : str, delimiter : str = None, verbose : bool = False):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        a/1.txt
        a/b/2.txt

    If you just specify prefix = 'a', you'll get back:

        a/1.txt
        a/b/2.txt

    However, if you specify prefix='a' and delimiter='/', you'll get back:

        a/1.txt

    Additionally, the same request will return blobs.prefixes populated with:

        a/b/
    """

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    if verbose:
        print("Blobs:")
        for blob in blobs:
            print(blob.name)

        if delimiter:
            print("Prefixes:")
            for prefix in blobs.prefixes:
                print(prefix)

    return blobs


# TODO:  Add return type hint
def explicit_list_blobs_with_prefix(bucket_name : str, service_account_key_file : str,
                                    prefix : str, delimiter : str = None, verbose : bool = False):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        a/1.txt
        a/b/2.txt

    If you just specify prefix = 'a', you'll get back:

        a/1.txt
        a/b/2.txt

    However, if you specify prefix='a' and delimiter='/', you'll get back:

        a/1.txt

    Additionally, the same request will return blobs.prefixes populated with:

        a/b/
    """

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    if verbose:
        print("Blobs:")
        for blob in blobs:
            print(blob.name)

        if delimiter:
            print("Prefixes:")
            for prefix in blobs.prefixes:
                print(prefix)

    return blobs


def download_blob(bucket_name : str, source_blob_name : str, destination_file_name : str):
    """Downloads a blob from the bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def explicit_download_blob(bucket_name : str, source_blob_name : str, destination_file_name : str,
                           service_account_key_file : str):
    """Downloads a blob from the bucket."""

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def delete_blob(bucket_name : str, blob_name : str):
    """Deletes a blob from the bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print("Blob {} deleted.".format(blob_name))


def explicit_delete_blob(bucket_name : str, blob_name : str, service_account_key_file : str):
    """Deletes a blob from the bucket."""

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print("Blob {} deleted.".format(blob_name))


def upload_blob(bucket_name : str, source_file_name : str, destination_blob_name : str, type : str = 'from_filename'):
    """Uploads a file to the bucket."""

    assert type in ['from_file', 'frim_filename', 'from_string']

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    if type == 'from_filename':
        blob.upload_from_filename(source_file_name)
    elif type == 'from_file':
        with open(source_file_name, "rb") as source_file:
            blob.upload_from_file(source_file)
    else:
        pass

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def explicit_upload_blob(bucket_name : str, source_file_name : str, destination_blob_name : str,
                         service_account_key_file : str, type : str = 'from_filename'):
    """Uploads a file to the bucket."""

    assert type in ['from_file', 'frim_filename', 'from_string']

    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(service_account_key_file)

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    if type == 'from_filename':
        blob.upload_from_filename(source_file_name)
    elif type == 'from_file':
        with open(source_file_name, "rb") as source_file:
            blob.upload_from_file(source_file)
    else:
        pass

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def streaming_upload_blob():
    # TODO: cannot currently perform streaming uploads with the Python client library.
    #  Need to try using gsutil API as follows:
    #  Pipe the data to the gsutil cp command and use a dash for the source URL:
    #  PROCESS_NAME | gsutil cp - gs://BUCKET_NAME/OBJECT_NAME
    #  Where:
    #  PROCESS_NAME is the name of the process from which you are collecting data. For example, collect_measurements.
    #  BUCKET_NAME is the name of the bucket containing the object. For example, my_app_bucket.
    #  OBJECT_NAME is the name of the object that is created from the data. For example, data_measurements.
    pass


def streaming_download_blob():
    # TODO: cannot currently perform streaming downloads with the Python client library.
    #  Need to try using gsutil API as follows:
    #  Run the gsutil cp command using a dash for the destination URL, then pipe the data to the process:
    #  gsutil cp gs://BUCKET_NAME/OBJECT_NAME - | PROCESS_NAME
    #  Where:
    #  BUCKET_NAME is the name of the bucket containing the object. For example, my_app_bucket.
    #  OBJECT_NAME is the name of the object that you are streaming to the process. For example, data_measurements.
    #  PROCESS_NAME is the name of the process into which you are feeding data. For example, analyze_data.
    pass

