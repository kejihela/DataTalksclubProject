from time import sleep
from prefect_aws import S3Bucket, AwsCredentials



def aws_credentials():
    s3_block_cred = AwsCredentials(
        aws_access_key_id="1234",
        aws_secret_access_key="************",
        region_name="eu-north-1"

    )
    s3_block_cred.save(name="my-aws-creds", overwrite = True)


def s3_bucket_block():
    aws_cred = AwsCredentials.load("my-aws-creds")
    s3_block_obj = S3Bucket(bucket_name="mlop-zoomcamp-adebayo", credentials = aws_cred)
    s3_block_obj.save(name="my-s3-bucket", overwrite = True )


if __name__ =="__main__":
    aws_credentials()
    sleep(5)
    s3_bucket_block()