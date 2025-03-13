#!/bin/bash
# This script repeatedly calls the faucet command to mint tokens for both
# the miner_f and validator_f wallets until each wallet's balance reaches 1000 taos.

TARGET=1000
CHAIN_ENDPOINT="ws://127.0.0.1:9944"

# Function to get the balance for a wallet.
# Assumes the balance output is in the form: "Balance: τ 1234.5678"
get_balance() {
    local wallet_name="$1"
    balance=$(btcli wallet balance --wallet.name "$wallet_name" --subtensor.chain_endpoint "$CHAIN_ENDPOINT" | grep -Eo '[0-9]+\.[0-9]+')
    echo "$balance"
}

while true; do
    miner_balance=$(get_balance miner_f)
    validator_balance=$(get_balance validator_f)
    
    echo "Current balances: miner_f = $miner_balance, validator_f = $validator_balance"
    
    # Check if both wallets have reached the target balance.
    if (( $(echo "$miner_balance >= $TARGET" | bc -l) )) && (( $(echo "$validator_balance >= $TARGET" | bc -l) )); then
        echo "Both wallets have reached at least $TARGET taos. Exiting."
        break
    fi
    
    # Mint tokens for miner_f if its balance is below TARGET.
    if (( $(echo "$miner_balance < $TARGET" | bc -l) )); then
        echo "Minting tokens for miner_f..."
        btcli wallet faucet --wallet.name miner_f --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
    fi
    
    # Mint tokens for validator_f if its balance is below TARGET.
    if (( $(echo "$validator_balance < $TARGET" | bc -l) )); then
        echo "Minting tokens for validator_f..."
        btcli wallet faucet --wallet.name validator_f --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
    fi
    
    # Wait for 10 seconds before checking the balances again.
    sleep 10
done
