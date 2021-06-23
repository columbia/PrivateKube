#!/bin/sh

# kubectl apply -f private-data-block.yaml
# kubectl apply -f privacy-budget-claim.yaml

# kubectl create namespace privacy-example
kubectl apply -f add-block_rdp_b.yaml


for i in 1 2 3
do
    sleep 2 

    kubectl apply -f add-claim-${i}b.yaml

    sleep 5

    # check the status
    kubectl describe privatedatablocks --namespace=privacy-example

    sleep 2 

    kubectl describe privacybudgetclaims --namespace=privacy-example

done
sleep 2

echo "\n\nFinal block state\n\n"
kubectl describe privatedatablocks --namespace=privacy-example


# clear pod
for i in 1 2 3
do
    kubectl delete -f add-claim-${i}b.yaml
done
kubectl delete -f add-block_rdp_b.yaml
# kubectl delete namespace privacy-example
