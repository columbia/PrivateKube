#!/bin/sh

# kubectl apply -f private-data-block.yaml
# kubectl apply -f privacy-budget-claim.yaml

# kubectl create namespace privacy-example
kubectl apply -f add-block.yaml
echo "\n\n\nInitial state of the block:"
kubectl describe privatedatablocks --namespace=privacy-example

sleep 2 

kubectl apply -f add-claim-1.yaml




sleep 10

# check the status

echo "\n\n\nFinal state of the block:"
kubectl describe privacybudgetclaims --namespace=privacy-example

sleep 5
echo "\n\n\nFinal state of the claim:"
kubectl describe privacybudgetclaims --namespace=privacy-example


# clear pod
kubectl delete -f add-claim-1.yaml
kubectl delete -f add-block.yaml
# kubectl delete namespace privacy-example
