#!/bin/bash

# Définir les autres arguments de la commande
input_file="../../Ressources/In/gecko.ppm"
filter_type="bilateral"
sigma_spatial="-s 5"
sigma_tonal="-t 120"

# # Boucle pour faire varier sigmaGaussian de 10 à 100 par pas de 10
# for sigma_gaussian in {1..100..1}
# do
#     # echo "Running with --sigmaGaussian=$sigma_gaussian"
#     ./main "$input_file" "$filter_type" "$sigma_spatial" "$sigma_tonal" --sigmaGaussian "$sigma_gaussian"
# done

# # Boucle pour faire varier sigmaSaltAndPepper de 0.001 à 0.1 par pas de 0.001
# sigma_salt_and_pepper=0.001
# while (( $(echo "$sigma_salt_and_pepper <= 0.1" | bc -l) )); do
#     ./main "$input_file" "$filter_type" "$sigma_spatial" "$sigma_tonal" --sigmaSaltAndPepper "$sigma_salt_and_pepper"
    
#     # Incrémenter sigma_salt_and_pepper de 0.01
#     sigma_salt_and_pepper=$(echo "$sigma_salt_and_pepper + 0.001" | bc)
# done

# # Boucle pour faire varier sigmaSpeckle de 0.02 à 2.0 par pas de 0.02
# sigma_speckle=0.005
# while (( $(echo "$sigma_speckle <= 0.5" | bc -l) )); do
#     ./main "$input_file" "$filter_type" "$sigma_spatial" "$sigma_tonal" --sigmaSpeckle "$sigma_speckle"
    
#     # Incrémenter sigma_speckle de 0.005
#     sigma_speckle=$(echo "$sigma_speckle + 0.005" | bc)
# done

# # Boucle pour faire varier sigma2 de 1 à 200 par pas de 1
# sigma_tonal="1"
# while (( $(echo "$sigma_tonal <= 200" | bc -l) )); do
#     ./main "$input_file" "$filter_type" "$sigma_spatial" -t "$sigma_tonal"
    
#     # Incrémenter sigma_tonal de 1
#     sigma_tonal=$(echo "$sigma_tonal + 1" | bc)
# done

# # Boucle pour faire varier sigma de 0.02 à 2 par pas de 0.02
# sigma_spatial="0.02"
# while (( $(echo "$sigma_spatial <= 2" | bc -l) )); do
#     ./main "$input_file" "$filter_type" -s "$sigma_spatial" "$sigma_tonal"
    
#     # Incrémenter sigma_spatial de 0.02
#     sigma_spatial=$(echo "$sigma_spatial + 0.02" | bc)
# done

# # Boucle pour faire varier sigma de 0.02 à 2 par pas de 0.02
# sigma_spatial="0.02"
# while (( $(echo "$sigma_spatial <= 2" | bc -l) )); do
#     ./main "$input_file" "$filter_type" -s "$sigma_spatial" "$sigma_tonal"
    
#     # Incrémenter sigma_spatial de 0.02
#     sigma_spatial=$(echo "$sigma_spatial + 0.02" | bc)
# done

sigma_gaussian="1"
sigma_speckle="0.02"
sigma_salt_and_pepper="0.001"

while (( $(echo "$sigma_gaussian <= 100" | bc -l) )); do
    ./main "$input_file" "$filter_type" "$sigma_spatial" "$sigma_tonal" --sigmaGaussian "$sigma_gaussian" --sigmaSaltAndPepper "$sigma_salt_and_pepper" --sigmaSpeckle "$sigma_speckle"
    
    sigma_gaussian=$(echo "$sigma_gaussian + 1" | bc)
    sigma_salt_and_pepper=$(echo "$sigma_salt_and_pepper + 0.001" | bc)
    sigma_speckle=$(echo "$sigma_speckle + 0.02" | bc)
done