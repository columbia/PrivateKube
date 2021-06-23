#!/bin/sh

# kubectl apply -f private-data-block.yaml
# kubectl apply -f privacy-budget-claim.yaml

# kubectl create namespace privacy-example
kubectl apply -f add-block_rdp.yaml

sleep 2 

kubectl apply -f add-claim-1.yaml

sleep 5

# check the status
kubectl describe privatedatablocks --namespace=privacy-example

sleep 2 

kubectl describe privacybudgetclaims --namespace=privacy-example

sleep 2

# clear pod
kubectl delete -f add-claim-1.yaml
kubectl delete -f add-block_rdp_b.yaml
# kubectl delete namespace privacy-example
