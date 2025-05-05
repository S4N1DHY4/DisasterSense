/* eslint-disable */
import {default as web3} from '../web3'

const address = 'your_contract_address';
const abi = [
    {
        "anonymous": false,
        "inputs": [
            {
                "indexed": false,
                "internalType": "address",
                "name": "causeAddress",
                "type": "address"
            },
            {
                "indexed": false,
                "internalType": "address",
                "name": "creator",
                "type": "address"
            },
            {
                "indexed": false,
                "internalType": "string",
                "name": "title",
                "type": "string"
            },
            {
                "indexed": false,
                "internalType": "string",
                "name": "cause_type",
                "type": "string"
            },
            {
                "indexed": false,
                "internalType": "string",
                "name": "desciption",
                "type": "string"
            },
            {
                "indexed": false,
                "internalType": "uint256",
                "name": "goal",
                "type": "uint256"
            }
        ],
        "name": "CauseCreated",
        "type": "event"
    },
    {
        "constant": true,
        "inputs": [],
        "name": "getAllCauses",
        "outputs": [
            {
                "internalType": "contract Cause[]",
                "name": "",
                "type": "address[]"
            }
        ],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {
                "internalType": "string",
                "name": "title",
                "type": "string"
            },
            {
                "internalType": "string",
                "name": "cause_type",
                "type": "string"
            },
            {
                "internalType": "string",
                "name": "description",
                "type": "string"
            },
            {
                "internalType": "uint256",
                "name": "goal",
                "type": "uint256"
            }
        ],
        "name": "startCause",
        "outputs": [],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    }
];
const DeSenseRescueInstance = new web3.eth.Contract(abi, address);

export default DeSenseRescueInstance;
