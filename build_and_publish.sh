# docker build dockerbuild -t k8simage
TAG=axlearn_neuronx-czhenguo
MLFLOW_TRACKING_URI="https://alpha.mlflow.neuron.annapurna.aws.dev"

docker build \
    --build-arg MLFLOW_TRACKING_URI_ARG=$MLFLOW_TRACKING_URI \
    docker -t $TAG

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-2
REPO=eks_trn2_axlearn

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker tag $TAG $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
# aws ecr create-repository --repository-name $REPO
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
