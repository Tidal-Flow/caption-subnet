#!/bin/bash
# This script repeatedly calls the faucet command for both miner_f and validator_f
# until each wallet's balance reaches at least 1000 taos.

TARGET=1000
CHAIN_ENDPOINT="ws://127.0.0.1:9944"

# Function to extract the balance (after the arrow) from btcli wallet overview output.
get_balance() {
    local wallet_name="$1"
    btcli wallet overview --wallet.name "$wallet_name" --subtensor.chain_endpoint "$CHAIN_ENDPOINT" --no_prompt \
    | grep 'Balance:' \
    | awk -F'➡' '{print $2}' \
    | awk '{print $1}' \
    | tr -d 'τ,'
}

while true; do
    miner_balance=$(get_balance miner_f)
    validator_balance=$(get_balance validator_f)

    echo "Current balances: miner_f = $miner_balance, validator_f = $validator_balance"

    # Check if both balances have reached the target
    if (( $(echo "$miner_balance >= $TARGET" | bc -l) )) && (( $(echo "$validator_balance >= $TARGET" | bc -l) )); then
        echo "Both wallets have reached $TARGET taos. Exiting."
        break
    fi

    # If miner_f is below target, run its faucet command
    if (( $(echo "$miner_balance < $TARGET" | bc -l) )); then
        printf "y\n" | btcli wallet faucet --wallet.name miner_f --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
    fi

    # If validator_f is below target, run its faucet command
    if (( $(echo "$validator_balance < $TARGET" | bc -l) )); then
        printf "y\n" | btcli wallet faucet --wallet.name validator_f --subtensor.chain_endpoint "$CHAIN_ENDPOINT"
    fi

    # Pause for 10 seconds before the next iteration
    sleep 10
done
