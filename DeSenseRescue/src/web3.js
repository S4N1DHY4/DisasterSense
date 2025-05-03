
import Web3 from 'web3';

if (window.ethereum) {
    window.web3 = new Web3(ethereum);
    try {
        
        ethereum.enable();
    } catch (error) {
    }
} else if (window.web3) {
    window.web3 = new Web3(web3.currentProvider);
} else {
    console.error('Non-Ethereum browser detected. You should consider trying MetaMask!');

}

export default window.web3;

