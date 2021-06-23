#!/bin/bash

kubectl create ns privatekube

kubectl apply -f system/privacyresource/artifacts/private-data-block.yaml

kubectl apply -f system/privacyresource/artifacts/privacy-budget-claim.yaml

kubectl apply -f system/dpfscheduler/manifests/cluster-role-binding.yaml

kubectl apply -f  system/dpfscheduler/manifests/scheduler.yaml
