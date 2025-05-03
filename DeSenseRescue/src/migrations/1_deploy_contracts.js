const DeSenseRescue = artifacts.require("DeSenseRescue");

module.exports = function(deployer) {
  deployer.deploy(DeSenseRescue, "your_wallet_address");
};
