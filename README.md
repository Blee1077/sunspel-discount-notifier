# Sunspel Riveria Polo-shirt Discount Notifier
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/you-didnt-ask-for-this.svg)](https://forthebadge.com)


The purpose of this project is to scrape the prices of Sunspel's Riviera polo shirts (https://www.sunspel.com/uk/mens/polo-shirts.html) on a daily basis and send me an email when there is a discount.  The core functionality is an AWS Lambda function that is scheduled to run on a daily basis using an EventBridge rule, AWS SAM is used to create all the necessary AWS resources to get this application up and running.

This repo contains source code and supporting files for a serverless application that can be deploy with the SAM CLI. It includes the following files and folders.

- sunspel-discount-notifier - Code for the application's Lambda function.
- template.yaml - A template that defines the application's AWS resources.

The application uses several AWS resources, including Lambda functions, an EventBridge rule, and an SNS topic. These resources are defined in the `template.yaml` file in this project.

## Pre-requisites

1. An AWS account
2. SAM CLI - [Install the SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
3. Python 3 - [Install Python 3](https://www.python.org/downloads/)
4. Docker - [Install Docker community edition](https://hub.docker.com/search/?type=edition&offering=community)
5. An pre-existing S3 bucket
6. A JSON file named containing email details with the following structure:

    ```yaml
    {
        "sender_email": "example1@mail.com"    // Address to send emails from
        "password": "passwordvalue"            // Password for sender_email
        "host": "mail.example.com"             // Email host server
        "port": 587                            // Port number needed to communicate with host server
        "receiver_email": "example2@mail.com"  // Address to receive emails (can be same as sender_email)
    }
    ```

## Configurations
The following steps need to be taken in the `template.yaml` file before the application can be deployed:

1. Replace the value of the `EMAIL_SECRET_BUCKET` global environment variable with your pre-existing S3 bucket name.
2. Similarly, replace the value of the `EMAIL_SECRET_JSON_KEY` global environment variable with the name of your JSON file containing email details placed in the root of your pre-existing S3 bucket.
3. (Optional) Adjust the value of `MIN_DISCOUNT_PERC` to your preference, by default this is set to 50% meaning that an email will only be sent once a product reaches a discount of 50%.

## Deploy the application

The Serverless Application Model Command Line Interface (SAM CLI) is an extension of the AWS CLI that adds functionality for building and testing Lambda applications. It uses Docker to run your functions in an Amazon Linux environment that matches Lambda. It can also emulate your application's build environment and API.

To build and deploy the application for the first time, run the following in your shell:

```bash
sam build
sam deploy --guided
```

The first command will build the source of this application. The second command will package and deploy this application to AWS, with a series of prompts:

* **Stack Name**: The name of the stack to deploy to CloudFormation. This should be unique to your account and region, and a good starting point would be something matching this project's function.
* **AWS Region**: The AWS region you want to deploy this app to.
* **SNS Email Parameter**: The email address to send execution failure notifications.
* **Confirm changes before deploy**: If set to yes, any change sets will be shown to you before execution for manual review. If set to no, the AWS SAM CLI will automatically deploy application changes.
* **Allow SAM CLI IAM role creation**: Many AWS SAM templates, including this one, create AWS IAM roles required for the AWS Lambda function(s) included to access AWS services. By default, these are scoped down to minimum required permissions. To deploy an AWS CloudFormation stack which creates or modifies IAM roles, the `CAPABILITY_IAM` value for `capabilities` must be provided. If permission isn't provided through this prompt, to deploy this example you must explicitly pass `--capabilities CAPABILITY_IAM` to the `sam deploy` command.
* **Save arguments to samconfig.toml**: If set to yes, your choices will be saved to a configuration file inside the project, so that in the future you can just re-run `sam deploy` without parameters to deploy changes to this application.

## Use the SAM CLI to build locally

Build this application with the `sam build` command.

```bash
sunspel-discount-notifier$ sam build
```

The SAM CLI installs dependencies defined in `sunspel-discount-notifier/requirements.txt`, creates a deployment package, and saves it in the `.aws-sam/build` folder.

## Cleanup

To delete the deployed application, use the AWS CLI. Assuming you used this project name for the stack name, you can run the following:

```bash
aws cloudformation delete-stack --stack-name sunspel-discount-notifier
```
