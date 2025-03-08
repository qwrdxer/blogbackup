---
title: hardhat智能合约流程
categories:
  - 区块链
date: 2024-01-21 15:20:03
tags:
---



安装包

```shell
// 必要的依赖
proxychains4 yarn add --dev @nomiclabs/hardhat-ethers@npm:hardhat-deploy-ethers ethers @nomiclabs/hardhat-etherscan @nomiclabs/hardhat-waffle @nomicfoundation/hardhat-verify chai@4 ethereum-waffle hardhat hardhat-contract-sizer hardhat-deploy hardhat-gas-reporter prettier prettier-plugin-solidity solhint solidity-coverage dotenv
proxychains4 yarn add --dev @nomicfoundation/hardhat-ethers@^3.0.2 
//项目的第三方依赖
proxychains4 yarn add --dev @chainlink/contracts



// verify 的升级
https://hardhat.org/hardhat-runner/docs/advanced/migrating-from-hardhat-waffle
yarn add --dev @nomicfoundation/hardhat-toolbox @nomicfoundation/hardhat-network-helpers @nomicfoundation/hardhat-chai-matchers@1 @nomiclabs/hardhat-ethers @nomiclabs/hardhat-etherscan chai ethers@5 hardhat-gas-reporter solidity-coverage @typechain/hardhat typechain @typechain/ethers-v6

```

初始化

```bash
yarn hardhat init 
```

编写hardhat.config.js



cp -r ../hardhat-smartcontract-lottery-fcc/deploy ../hardhat-smartcontract-lottery-fcc/contracts/ ../hardhat-smartcontract-lottery-fcc/utils/ ../hardhat-smartcontract-lottery-fcc/.env ../hardhat-smartcontract-lottery-fcc/helper-hardhat-config.js    ../hardhat-smartcontract-lottery-fcc/hardhat.config.js .

 

```js
require("@nomiclabs/hardhat-waffle")
require("@nomiclabs/hardhat-etherscan")
require("hardhat-deploy")
require("solidity-coverage")
require("hardhat-gas-reporter")
require("hardhat-contract-sizer")
require("dotenv").config()


const SEPOLIA_RPC_URL = process.env.SEPOLIA_RPC_URL
const PRIVATE_KEY = process.env.PRIVATE_KEY
const COINMARKETCAP_API_KEY = process.env.COINMARKETCAP_API_KEY
const ETHERSCAN_API_KEY = process.env.ETHERSCAN_API_KEY
const REPORT_GAS = process.env.REPORT_GAS || false
/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
    defaultNetwork: "hardhat",
    networks: {
        hardhat: {
            chainId: 31337,
            blockConfirmations: 1,
        },
        sepolia: {
            chainId: 11155111,
            blockConfirmations: 1,
            url: SEPOLIA_RPC_URL,
            accounts: [PRIVATE_KEY],
        },
    },
    solidity: "0.8.7",
    namedAccounts: {
        deployer: {
            default: 0, // here this will by default take the first account as deployer
            1: 0, // similarly on mainnet it will take the first account as deployer. Note though that depending on how hardhat network are configured, the account 0 on one network can be different than on another
        },
        player: {
            default: 1,
        },
    },
    etherscan: {
        apiKey: ETHERSCAN_API_KEY,
    },
    gasReporter: {
        enabled: REPORT_GAS,
        currency: "USD",
        outputFile: "gas-report.txt",
        noColors: true,
        // coinmarketcap: process.env.COINMARKETCAP_API_KEY,
    },
}

```

编写.env文件

```bash
SEPOLIA_RPC_URL='https://eth-sepolia.g.alchemy.com/v2/ySrtsutceOnTTsxp9TiP9f'
POLYGON_MAINNET_RPC_URL='https://rpc-mainnet.maticvigil.com'
PRIVATE_KEY='5b87f02d56cee5838afa0a1c47c29e456afa763657deb4d614044b'
ALCHEMY_MAINNET_RPC_URL="SQ-R3S6RCf5yD-kwgfyk_1qBrm"
REPORT_GAS=true
ETHERSCAN_API_KEY="6HQIA6UXF1N8J41XYCCZ27JQN"
COINMARKETCAP_API_KEY="5f416d32-f038-4a4c-97ad-6b1a"
AUTO_FUND=true
```



编写helper-hardhat-config.js

```js
const { ethers } = require("hardhat")

const networkConfig = {
    11155111: {
        name: "sepolia",
        vrfCoordinatorV2: "0x8103B0A8A00be2DDC778e6e7eaa21791Cd364625",
        entranceFee: ethers.parseEther("0.01"),
        //https://docs.chain.link/vrf/v2/subscription/supported-networks
        gasLane: "0x474e34a077df58807dbe9c96d3c009b23b3c6d0cce433e59bbf5b34f823bc56c",
        subscriptionId: "0",
        callbackGasLimit: "500000",
        interval: "30",
    },
    31337: {
        name: "hardhat",
        entranceFee: ethers.parseEther("0.01"),
        gasLane: "0x474e34a077df58807dbe9c96d3c009b23b3c6d0cce433e59bbf5b34f823bc56c",
        callbackGasLimit: "500000",
        interval: "30",
    },
}
const developmentChains = ["hardhat", "localhost"]
//导出模块
module.exports = {
    networkConfig,
    developmentChains,
}

```

创建 contracts  编写合约

创建 deploy  编写部署脚本

创建 test  编写测试脚本

---

的文章参考:

博客地址: qwrdxer.github.io

欢迎交流: qq1944270374