#!/bin/bash
# Bash script för att testa API:et (om du har Git Bash eller WSL)

echo "Testing Avarn ML API"
echo "===================="

# 1. Health Check
echo ""
echo "1. Health Check:"
curl http://localhost:8000/health

# 2. Single Prediction
echo ""
echo ""
echo "2. Making Prediction:"
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "antal_sektioner": 8,
    "antal_detektorer": 25,
    "antal_larmdon": 15,
    "dörrhållarmagneter": 5,
    "ventilation": 1,
    "stad": "Stockholm",
    "kvartalsvis": 0,
    "månadsvis": 1,
    "årsvis": 0
  }'

echo ""
echo ""
echo "Done! Open http://localhost:8000/docs for Swagger UI"
