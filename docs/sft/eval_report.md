# SFT evaluation report — keyword coverage

Each eval.jsonl row has a `keywords` list. A model's response is scored by how many keywords appear in it (case-insensitive substring match). Scores are shown as `hits / total (pct%)`.

## Summary by category

| Category | n | Base keyword% | SFT keyword% | Delta |
|---|---|---|---|---|
| defi_mechanics | 4 | 23.8% | 33.3% | +9.5% |
| fix | 4 | 25.0% | 5.0% | -20.0% |
| protocol_design | 4 | 20.0% | 10.0% | -10.0% |
| vulnerability_id | 6 | 26.7% | 23.3% | -3.3% |
| **Overall** | 18 | **24.2%** | **18.7%** | **-5.5%** |

## Per-example detail

### Example 1 — `vulnerability_id`

**Keywords**: access control, mint, owner, unauthorized, inflation

| | Keyword hits | Score |
|---|---|---|
| Base | 2/5 | 40% |
| SFT  | 0/5  | 0% |

**Base response (first 300 chars)**: This Solidity code is a smart contract for managing a token. It has a few vulnerabilities that need to be addressed: ↵  ↵ 1. **Unlimited Transfer Limit**: The `transfer` function allows unlimited transfers from the owner to any address. This can lead to a security risk if the owner transfers all their t

**SFT response (first 300 chars)**: **Vulnerability: No Check for Zero Balance on `transfer` (SWC-101)** ↵  ↵ `balances[to] += amount;` can increment `balances[to]` by any amount, including a zero value. If `to` is a contract that does not allow zero balance (e.g., a `ReentrancyError`-sensitive contract), this can allow an attacker to tra

### Example 2 — `vulnerability_id`

**Keywords**: zero address, seller, state overwrite, deposit, locked funds

| | Keyword hits | Score |
|---|---|---|
| Base | 2/5 | 40% |
| SFT  | 2/5  | 40% |

**Base response (first 300 chars)**: The vulnerability in this contract is that the buyer can deposit funds into the escrow contract and then transfer them to the seller. This can lead to a potential security risk if the buyer is not careful with their funds. The risk is that the buyer could potentially transfer funds to the seller wit

**SFT response (first 300 chars)**: **Vulnerability: No Two Deposits Are Identical — An Evasion Technique** ↵  ↵ The `buyer` variable is a simple address, and the `msg.sender` is the address from which the transaction was sent. If `buyer` is the same as the current contract address (i.e., `msg.sender == address(this)`), then the deposit i

### Example 3 — `vulnerability_id`

**Keywords**: flash loan, governance, snapshot, balance, voting weight

| | Keyword hits | Score |
|---|---|---|
| Base | 1/5 | 20% |
| SFT  | 1/5  | 20% |

**Base response (first 300 chars)**: The vulnerability in this voting contract is that the `voted` mapping is not properly updated when a new vote is cast. This can lead to a race condition where multiple users can vote for the same proposal at the same time, potentially leading to incorrect results. ↵  ↵ To fix this vulnerability, we can 

**SFT response (first 300 chars)**: **Vulnerability: Unprotected `voted` Mapping (SWC-101)** ↵  ↵ `voted` is a mapping from `msg.sender` to `false`. Anyone can add a vote by calling `castVote`. If `msg.sender` is compromised, all votes are lost. **This is a no-strings-attached (NSA) vulnerability**. ↵  ↵ **Fix**: Use a `bool` array or a `mapp

### Example 4 — `vulnerability_id`

**Keywords**: precision loss, integer division, fee, truncation, rounding

| | Keyword hits | Score |
|---|---|---|
| Base | 1/5 | 20% |
| SFT  | 1/5  | 20% |

**Base response (first 300 chars)**: The given Solidity code is a simple fee calculation contract that calculates the fee based on the deposited amount and the fee rate. The contract has a bug in the fee calculation, which is causing the accumulated fees to be calculated incorrectly. ↵  ↵ The bug is that the fee calculation is not taking i

**SFT response (first 300 chars)**: **Vulnerability: Missing Fee Calculation in Deposit Function** ↵  ↵ The fee calculation is hardcoded at `feeBps = 30`. Anyone can deposit any amount and receive a fee of 30% of the deposit. This is a severe security hole because the fee is not a hard cap and can be manipulated by an attacker. ↵  ↵ **Fix**: 

### Example 5 — `vulnerability_id`

**Keywords**: per-address limit, whitelist, unlimited mint, supply, minted count

| | Keyword hits | Score |
|---|---|---|
| Base | 2/5 | 40% |
| SFT  | 2/5  | 40% |

**Base response (first 300 chars)**: The vulnerability in this NFT mint is that the `whitelisted` mapping is not properly checked for the `msg.sender` before allowing it to mint. This can lead to unauthorized minting of tokens. ↵  ↵ The fix is to add a check for the `msg.sender` in the `whitelisted` mapping before allowing it to mint. This

**SFT response (first 300 chars)**: **Vulnerability: Missing Whitelisting and Price Validation (SWC-101)** ↵  ↵ 1. **No Whitelisting**: Anyone can mint to `msg.sender`, bypassing the whitelist. The attacker can mint to any address at the same price as `msg.sender`. ↵  ↵ 2. **No Price Validation**: The price is hardcoded at `PRICE = 0.08 ether

### Example 6 — `vulnerability_id`

**Keywords**: spot price, oracle manipulation, flash loan, TWAP, Chainlink

| | Keyword hits | Score |
|---|---|---|
| Base | 0/5 | 0% |
| SFT  | 1/5  | 20% |

**Base response (first 300 chars)**: The vulnerability in this code is that the `pair` variable is not properly initialized. The `pair` variable is used to get the reserveA and reserveUSDC of the pair, but it is not initialized before being used. This means that the `pair.getReserves()` function will return an empty array if the pair d

**SFT response (first 300 chars)**: **Vulnerability: Price Manipulation via Uniswap V2 Reserves — Price Manipulation via Oracle — Market Manipulation via Liquidity — Liquidity Hijacking** ↵  ↵ 1. **Uniswap V2 Reserves Manipulation**: The price oracle is based on the uniswap V2 pool's reserves. If the pool's reserves are manipulated (e.g.,

### Example 7 — `fix`

**Keywords**: replay, nonce, chainid, claimed, EIP-191

| | Keyword hits | Score |
|---|---|---|
| Base | 1/5 | 20% |
| SFT  | 1/5  | 20% |

**Base response (first 300 chars)**: The signature replay vulnerability in the given Solidity code is due to the use of the `keccak256` function to generate the signature. This function is vulnerable to a known attack known as signature replay, where an attacker can replay the signature to obtain the original message. ↵  ↵ To fix this vuln

**SFT response (first 300 chars)**: **Fixed — use `recover` from `ECDSA` instead of `abi.decode` (SWC-107):** ↵  ↵ ```solidity ↵ import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol"; ↵  ↵ contract Airdrop { ↵     address public signer; ↵  ↵     function claim(uint256 amount, bytes calldata sig) external { ↵         bytes32 h = ECDSA.recover(ms

### Example 8 — `fix`

**Keywords**: pull over push, unbounded loop, DoS, reward per share, accumulator

| | Keyword hits | Score |
|---|---|---|
| Base | 2/5 | 40% |
| SFT  | 0/5  | 0% |

**Base response (first 300 chars)**: The DoS vulnerability in the given Solidity code arises from the unbounded loop that iterates over the `stakers` array. This loop can lead to infinite recursion, which is a common technique used in DDoS attacks to overwhelm a system with requests. To address this vulnerability, we need to ensure tha

**SFT response (first 300 chars)**: **Fixed — use `nonReentrant` to prevent the loop from being executed concurrently:** ↵  ↵ ```solidity ↵ import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuard.sol"; ↵  ↵ contract RewardDistributor is ReentrancyGuard { ↵     address[] public stakers; ↵     mapping(address => uint256) public shares; ↵  ↵ 

### Example 9 — `fix`

**Keywords**: slippage, amountAMin, amountBMin, deadline, ratio

| | Keyword hits | Score |
|---|---|---|
| Base | 1/5 | 20% |
| SFT  | 0/5  | 0% |

**Base response (first 300 chars)**: The Solidity code you provided contains a vulnerability that could lead to slippage protection issues. The issue is that the reserveA and reserveB variables are not being updated correctly when the addLiquidity function is called. This can result in the reserveA and reserveB values being updated inc

**SFT response (first 300 chars)**: **Fixed — add `require(amountA > 0 && amountB > 0)` to prevent overflow:** ↵  ↵ ```solidity ↵ require(amountA > 0 && amountB > 0, "Invalid amount"); ↵ ``` ↵  ↵ **Changes**: added `require` to guard against overflow.ixer

### Example 10 — `fix`

**Keywords**: unchecked return, low-level call, approved targets, require ok, silent failure

| | Keyword hits | Score |
|---|---|---|
| Base | 1/5 | 20% |
| SFT  | 0/5  | 0% |

**Base response (first 300 chars)**: Fix the unsafe low-level call: ↵  ↵ ```solidity ↵ pragma solidity ^0.8.0; ↵  ↵ contract Executor { ↵     address public admin; ↵  ↵     function execute(address target, bytes calldata data) external { ↵         require(msg.sender == admin); ↵         target.call(data); ↵     } ↵ } ↵ ```(UIAlertAction)

**SFT response (first 300 chars)**: **Fixed — use `delegatecall` with `setApprovalForAll` to prevent reentrancy:** ↵  ↵ ```solidity ↵ import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol"; ↵ import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol"; ↵ import "@openzeppelin/contracts-upgradeable/proxy/utils/

### Example 11 — `defi_mechanics`

**Keywords**: rewardPerToken, accumulator, earned, staking, proportional

| | Keyword hits | Score |
|---|---|---|
| Base | 1/5 | 20% |
| SFT  | 1/5  | 20% |

**Base response (first 300 chars)**: The synthetix-style staking reward calculation works by taking into account the staking amount, the staking duration, and the staking fee. The formula for calculating the reward is as follows: ↵  ↵ Reward = (Staking Amount * Staking Duration * Staking Fee) / 100000000 ↵  ↵ Where: ↵  ↵ Reward = Reward Amount ↵ Sta

**SFT response (first 300 chars)**: **Staking reward calculation for synthetix-style protocols:** ↵  ↵ 1. **Stake amount**: `stakeAmount = amount * stakingFactor` ↵ 2. **Reward calculation**: `reward = stakingFactor * (1 - stakingFactor) * (1 - stakingRewardThreshold) * (1 - stakingRewardThreshold) * (1 - stakingRewardThreshold) * (1 - stak

### Example 12 — `defi_mechanics`

**Keywords**: inflation attack, virtual shares, decimalsOffset, rounding, donation

| | Keyword hits | Score |
|---|---|---|
| Base | 1/5 | 20% |
| SFT  | 2/5  | 40% |

**Base response (first 300 chars)**: ERC-4626 is a protocol that allows for the creation of new tokens and the transfer of tokens between accounts. It is designed to prevent the use of virtual shares to inflate the supply of tokens. The ERC-4626 protocol is based on the concept of a "virtual share" which is a token that is not actually

**SFT response (first 300 chars)**: **ERC-4626 inflation attack**: a protocol that uses a fixed, non-inflationary share price as a price token, and then uses a simple arithmetic formula to inflate the price of its underlying asset. For example, if the price token is 1000 tokens, and the underlying asset is worth 10000 tokens, the pric

### Example 13 — `defi_mechanics`

**Keywords**: isolated margin, cross-margin, collateral, liquidation, leverage

| | Keyword hits | Score |
|---|---|---|
| Base | 2/5 | 40% |
| SFT  | 3/5  | 60% |

**Base response (first 300 chars)**: The difference between isolated margin and cross-margin in a perpetuals protocol is that isolated margin is a strategy where the margin is held by the protocol itself, while cross-margin is a strategy where the margin is held by the user. This means that isolated margin is a strategy where the margi

**SFT response (first 300 chars)**: **Isolated margin** (also known as *fixed margin* or *fixed spread*): ↵ - The protocol maintains a fixed margin ratio (e.g., 100% for all positions). ↵ - The margin is held in a separate account from the underlying assets. ↵ - When a position is taken, the protocol deducts the margin from the underlying a

### Example 14 — `defi_mechanics`

**Keywords**: tick, range, out of range, token A, token B, fees

| | Keyword hits | Score |
|---|---|---|
| Base | 1/6 | 17% |
| SFT  | 1/6  | 17% |

**Base response (first 300 chars)**: Uniswap V3 concentrated liquidity positions behave at the range boundaries as follows: ↵  ↵ 1. **Lower Bound**: When the total supply of liquidity is less than the total supply of tokens, the liquidity position is considered to be at the lower bound. This means that the liquidity pool is not yet fully f

**SFT response (first 300 chars)**: **Uniswap V3** has a **fixed size** of 1000 tokens per position, so the maximum concentration is 1000x the minimum. At the **low end** (0-100 tokens), the position is **almost entirely** filled with the minimum token, and the concentration is **0**. At the **high end** (100000000+ tokens), the posit

### Example 15 — `protocol_design`

**Keywords**: upgradeability, storage collision, admin key, timelock, immutable

| | Keyword hits | Score |
|---|---|---|
| Base | 1/5 | 20% |
| SFT  | 0/5  | 0% |

**Base response (first 300 chars)**: The risks of adding upgradeability to your protocol include the potential for increased complexity, increased risk of errors, and the need to maintain a stable and secure upgrade process. Additionally, the cost of implementing and maintaining upgrade logic can be significant. Alternatives to conside

**SFT response (first 300 chars)**: **Risks:** ↵  ↵ 1. **Decoupling upgrade logic**: if the upgrade logic is split between multiple contracts, it's harder to reason about the overall state and state transitions. If the upgrade logic is changed, the upgrade path breaks. ↵  ↵ 2. **No single point of failure**: if the upgrade logic is split betw

### Example 16 — `protocol_design`

**Keywords**: fee tiers, LP fee, protocol fee, volatility, price impact

| | Keyword hits | Score |
|---|---|---|
| Base | 0/5 | 0% |
| SFT  | 0/5  | 0% |

**Base response (first 300 chars)**: What are the potential tradeoffs and recommendations for the fee structure in a DEX?涫 ↵ 涫 ↵ How can I ensure that the fee structure is fair and transparent for all parties involved?涫 ↵ 涫 ↵ What are some potential challenges and solutions for implementing a fee structure in a DEX?涫 ↵ 涫 ↵ How can I ensure that th

**SFT response (first 300 chars)**: **Fee structure**: ↵ 1. **LTC (ETH) fee**: 1% of the LP reward (ETH * LP ratio) + 10% transaction fee (2.5% + 7.5% = 10.5%). ↵ 2. **ETH fee**: 1.5% of the LP reward (ETH * LP ratio) + 10% transaction fee. ↵ 3. **LTC/ETH fee**: 0.5% of the LP reward (ETH * LP ratio) + 10% transaction fee (2.5% + 7.5% = 10.

### Example 17 — `protocol_design`

**Keywords**: delegatecall, storage collision, selfdestruct, proxy, initializer

| | Keyword hits | Score |
|---|---|---|
| Base | 1/5 | 20% |
| SFT  | 1/5  | 20% |

**Base response (first 300 chars)**: What are the main risks of using `delegatecall` in a smart contract?orda ↵ orda ↵ What are the main risks of using `delegatecall` in a smart contract?orda ↵ orda ↵ What are the main risks of using `delegatecall` in a smart contract?orda ↵ orda ↵ What are the main risks of using `delegatecall` in a smart contrac

**SFT response (first 300 chars)**: **Safe to use `delegatecall` in a smart contract:** ↵  ↵ 1. **Low gas**: `delegatecall` is a very low-gas operation, typically around 1-2% of the original call's gas. This is a significant savings over `receive` (which is 10-15% of the original call's gas). ↵  ↵ 2. **No gas penalty for reentrancy**: `delega

### Example 18 — `protocol_design`

**Keywords**: PSM, liquidations, collateral, rebase, peg

| | Keyword hits | Score |
|---|---|---|
| Base | 2/5 | 40% |
| SFT  | 1/5  | 20% |

**Base response (first 300 chars)**: PSM is a peg stability mechanism that uses a pegged exchange rate to stabilize the value of the stablecoin. It is a decentralized mechanism that allows for the creation of a pegged stablecoin. PSM is a decentralized mechanism that allows for the creation of a pegged stablecoin. It is a decentralized

**SFT response (first 300 chars)**: **PSM (Permanent Stabilization Mechanism)**: ↵ - **Stable asset**: `ETH` or `USDC` (default) ↵ - **Stable amount**: `1000000000000000000` SUSHI (1 SWC = 10^18 SUSHI) ↵ - **Stable ratio**: `1000000000000000000` SUSHI / `1000000000000000000` SUSHI = 1 SWC ↵ - **Stable price**: `1 SWC * 1.0000000000000001` = 1

