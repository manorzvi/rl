#!/bin/bash

echo "Welcome to Creating a service account Script!"

while getopts n: option
do
case "${option}"
in
n) NAME=${OPTARG};;
esac
done

if [ -n "$NAME" ]; then
    echo "NAME: $NAME"
else
    echo "Please specify service-account name before continue. $0 -n [service-account name]"
    exit
fi

PROJECT_ID=`gcloud config get-value project`
echo "PROJECT_ID: $PROJECT_ID"

echo "Create the service account ..."
X=`gcloud iam service-accounts create $NAME`
echo "Done."

echo "Grant permissions to the service account ..."
gcloud projects add-iam-policy-binding $PROJECT_ID --member "serviceAccount:$NAME@$PROJECT_ID.iam.gserviceaccount.com" --role "roles/owner"
echo "Done."

echo "Generate the key file ..."
gcloud iam service-accounts keys create $NAME.json --iam-account $NAME@$PROJECT_ID.iam.gserviceaccount.com
echo "Done. "

echo "Setting GOOGLE_APPLICATION_CREDENTIALS=$NAME.json env variable ..."
export GOOGLE_APPLICATION_CREDENTIALS="$NAME.json"
echo "Done."


