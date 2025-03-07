# docker build dockerbuild -t k8simage
if [[ -z "$PROJECT_PATH" ]]; then
    echo "PROJECT_PATH not set as env var!"
    exit
else
    echo "PROJECT_PATH set as ${PROJECT_PATH}"
fi

TAG=axlearn_neuronx
docker build --build-arg PROJECT_PATH_ARG=$PROJECT_PATH docker -t $TAG

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-2
REPO=eks_trn2_axlearn

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker tag $TAG $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
# aws ecr create-repository --repository-name $REPO
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
