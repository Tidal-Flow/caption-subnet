- ssh to machine
- conda activate subtensor
- source btsdk_venv/bin/activate
- cd subtensor
- pm2 start ./scripts/localnet.sh --name localnet --interpreter bash --env BUILD_BINARY=0

or if the pm2 instance exists or is active you start it

- pm2 list
- pm2 restart 0
- pm2 logs


`pm2 start python --name miner -- src/miner.py --netuid 3 --subtensor.chain_endpoint ws://127.0.0.1:9944 --wallet.name miner --wallet.hotkey default --logging.debug --axon.port 8091 --network local`