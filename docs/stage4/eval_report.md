# Stage 4 evaluation report — Base vs SFT vs DPO

Keyword coverage per model. Win-rate = DPO rows where DPO hits ≥ SFT hits.

## Summary by category

| Category | n | SFT | DPO | Δ (DPO − SFT) |
|---|---|---|---|---|
| defi_mechanics | 5 | 48.0% | 44.0% | -4.0 pp |
| fix | 2 | 10.0% | 10.0% | +0.0 pp |
| protocol_design | 4 | 33.3% | 33.3% | +0.0 pp |
| vulnerability_id | 4 | 15.0% | 20.0% | +5.0 pp |
| **Overall** | **15** | **30.3%** | **30.3%** | **+0.0 pp** |

## Win-rate: DPO vs SFT

| Metric | Count | Rate |
|---|---|---|
| DPO wins (DPO > SFT) | 2 | 13.3% |
| Draws (DPO = SFT)    | 11 | 73.3% |
| DPO losses (DPO < SFT) | 2 | 13.3% |

**Win-rate** (wins / total): **13.3%**  (excludes draws: 50.0%)

## Per-example responses

### Example 1 — `vulnerability_id`

**Prompt (first 200 chars)**: function withdraw(uint256 amount) external {
    require(balances[msg.sender] >= amount);
    (bool ok,) = msg.sender.call{value: amount}("");
    require(ok);
    balances[msg.sender] -= amount;
}

**Keywords**: reentrancy, checks-effects-interactions, external call, balance, CEI

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 1/5 | 20% | **Vulnerability: Unprotected Call to `call` (SWC-101)**  `call` is a system call that executes the caller's code on the target contract's address. If the contract does not allow external calls (e.g.,  |
| DPO | 2/5 | 40% | **Vulnerability: Unprotected Call to `call` (SWC-101)**  `call` is a system call that executes the caller's code on the target contract's address. If the contract does not allow external calls (e.g.,  |

### Example 2 — `vulnerability_id`

**Prompt (first 200 chars)**: function mint(address to, uint256 amount) external {
    _mint(to, amount);
}

**Keywords**: access control, onlyOwner, unauthorized, inflation, permission

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 0/5 | 0% | **Vulnerability: Missing `from` Parameter (SWC-101)**  The `from` parameter is required for `transfer` and `transferFrom`, but not for `mint`. An attacker can pass a `from` that is the contract itself |
| DPO | 0/5 | 0% | **Vulnerability: Missing `from` Parameter (SWC-101)**  The `from` parameter is required for `transfer` and `transferFrom`, but not for `mint`. An attacker can pass a `from` that is the contract itself |

### Example 3 — `vulnerability_id`

**Prompt (first 200 chars)**: function auth() external view returns (bool) {
    return tx.origin == owner;
}

**Keywords**: tx.origin, phishing, msg.sender, authentication, SWC-115

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 1/5 | 20% | **Vulnerability: Missing `tx.origin` Check (SWC-101)**  `tx.origin` is a `bytes32` constant that is always `0x0` on the first call to `tx.origin`. If the contract is deployed with a different address  |
| DPO | 1/5 | 20% | **Vulnerability: Missing `tx.origin` Check (SWC-101)**  `tx.origin` is a `bytes32` constant that is always `0x0` on the first call to `tx.origin`. If the contract is deployed with a different address  |

### Example 4 — `fix`

**Prompt (first 200 chars)**: function claim(uint256 amount, bytes calldata sig) external {
    bytes32 hash = keccak256(abi.encodePacked(msg.sender, amount));
    address signer = ECDSA.recover(hash, sig);
    require(signer == t

**Keywords**: replay, nonce, chainid, EIP-712, signature

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 1/5 | 20% | **Vulnerability: Missing Signature Verification (SWC-101)**  The `sig` parameter is a human-readable string that includes the `to` address, `value`, and `data`. The `recover` function is a cryptograph |
| DPO | 1/5 | 20% | **Vulnerability: Missing Signature Verification (SWC-101)**  The `sig` parameter is a human-readable string that includes the `to` address, `value`, and `data`. The `recover` function is a cryptograph |

### Example 5 — `fix`

**Prompt (first 200 chars)**: function addLiquidity(uint256 amountA, uint256 amountB) external {
    tokenA.transferFrom(msg.sender, address(this), amountA);
    tokenB.transferFrom(msg.sender, address(this), amountB);
    uint256

**Keywords**: slippage, amountMin, deadline, front-running, minimum

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 0/5 | 0% | **Vulnerability: Unprotected `transferFrom` — any address can transfer from `address(this)` — high risk of abuse**  `tokenA.transferFrom(msg.sender, address(this), amountA)` transfers `amountA` from ` |
| DPO | 0/5 | 0% | **Vulnerability: Unprotected `transferFrom` — any address can transfer from `address(this)` — high risk of abuse**  `transferFrom` is a public function that allows any address to transfer from `addres |

### Example 6 — `defi_mechanics`

**Prompt (first 200 chars)**: Explain the constant-product AMM formula x*y=k and how price impact works.

**Keywords**: constant product, price impact, reserves, slippage, k

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 2/5 | 40% | **Constant-Product (CP) Aggregator:**  `x * y = k` (CP formula)  **Price Impact**: - When `x` increases, `y` decreases (price drops). - When `y` increases, `x` decreases (price rises).  **Example**: - |
| DPO | 2/5 | 40% | **Constant-Product (CP) Aggregator:**  `x * y = k` (CP formula)  **Price Impact**: - When `x` increases, `y` decreases (CP effect). - When `y` increases, `x` decreases (CP effect). - The ratio `x / y` |

### Example 7 — `defi_mechanics`

**Prompt (first 200 chars)**: What is impermanent loss and when does it become permanent?

**Keywords**: impermanent loss, price ratio, liquidity, fees, withdraw

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 2/5 | 40% | **Impermanent loss (IL)** is the loss of value that occurs when a user's stake is reduced to a point where the protocol's liquidity pool is no longer sufficient to cover the loss. The IL is permanent  |
| DPO | 2/5 | 40% | **Impermanent loss (IL)** is the loss of value that occurs when a user's stake is reduced to a point where the protocol's liquidity pool is no longer sufficient to cover the loss. The IL is permanent  |

### Example 8 — `defi_mechanics`

**Prompt (first 200 chars)**: How does a flash loan work and what makes it dangerous with spot-price oracles?

**Keywords**: flash loan, single transaction, repay, oracle, manipulation

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 3/5 | 60% | **Flash loan**: a protocol that allows a user to borrow a large amount of a token (e.g., ETH) without collateral (e.g., a stablecoin) for a short period (e.g., 1-2 minutes). The protocol then repays t |
| DPO | 2/5 | 40% | **Flash loan**: a protocol that allows a user to borrow a large amount of a token (e.g., ETH) without collateral (e.g., a stablecoin) in exchange for a short-term loan (e.g., 1-24 hours) with a fixed  |

### Example 9 — `defi_mechanics`

**Prompt (first 200 chars)**: What is the health factor in Aave and how does liquidation work?

**Keywords**: health factor, liquidation threshold, collateral, borrow, undercollateralized

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 2/5 | 40% | **Health factor (H)** is a measure of the health of a contract. It is calculated as:  \[ H = \frac{1}{1 + e^{-rT}} \]  where: - \( r \) is the annualized return (e.g., 0.05 for Aave V2). - \( T \) is  |
| DPO | 2/5 | 40% | **Health factor (H)** is a measure of the health of a contract. It is calculated as:  \[ H = \frac{1}{1 + e^{-rT}} \]  where: - \( r \) is the annualized return (e.g., 0.05 for Aave V2). - \( T \) is  |

### Example 10 — `protocol_design`

**Prompt (first 200 chars)**: What are the main risks of upgradeable proxy contracts?

**Keywords**: storage collision, admin key, selfdestruct, delegatecall, timelock

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 0/5 | 0% | **Risks:**  1. **No state saving**: if the proxy is compromised, the caller's state is lost. This is a significant security hole in a decentralized application (DApp).  2. **No revert on invalid proxy |
| DPO | 0/5 | 0% | **Risks:**  1. **No state saving**: if the proxy is compromised, the caller's state is lost. This is a significant security hole in a decentralized application.  2. **No revert on invalid proxy calls* |

### Example 11 — `protocol_design`

**Prompt (first 200 chars)**: Compare UUPS vs Transparent proxy patterns.

**Keywords**: UUPS, transparent proxy, upgrade, implementation, ProxyAdmin

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 1/5 | 20% | **UUPS (Unbounded Scalability with Uniform Proxies)**: - **No gas penalty**: the proxy is always the same, so no gas is wasted on re-entrancy. - **No state storage**: the proxy is always the same, so  |
| DPO | 2/5 | 40% | **UUPS (Unbounded Scalability with Uniform Proxies)**: - **No gas penalty**: the proxy is always the same, so no gas is wasted on re-entrancy. - **No state storage**: the proxy is immutable and doesn' |

### Example 12 — `protocol_design`

**Prompt (first 200 chars)**: What is ERC-4626 and what is the inflation attack vector?

**Keywords**: ERC-4626, vault, shares, inflation attack, virtual shares

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 2/5 | 40% | **ERC-4626** is a standard ERC-2612 interface that allows for the creation of a token with a fixed supply and a non-zero mintable fee. The interface is defined as follows:  ```solidity interface ERC46 |
| DPO | 2/5 | 40% | **ERC-4626 (ERC-20 with inflation)** is a standard ERC-20 token that supports a floating price token. It uses a base token (e.g., USDT) and a secondary token (e.g., USDC) that is always worth 100% of  |

### Example 13 — `vulnerability_id`

**Prompt (first 200 chars)**: function getPrice() external view returns (uint256) {
    (uint112 r0, uint112 r1,) = IUniswapV2Pair(pair).getReserves();
    return r1 * 1e18 / r0;
}

**Keywords**: oracle manipulation, flash loan, TWAP, spot price, reserves

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 1/5 | 20% | **Vulnerability: Unchecked Return Value (SWC-107)**  `getReserves()` returns an `uint112` (20 digits) value, but `r1 * 1e18` is a `uint256` (512 digits). The multiplication is unchecked, and the resul |
| DPO | 1/5 | 20% | **Vulnerability: Unchecked Return Value (SWC-104)**  `getReserves()` returns an `uint112` (256 bits) value, but `r1 * 1e18` is a 64-bit integer. The multiplication is unchecked, and the result is lost |

### Example 14 — `defi_mechanics`

**Prompt (first 200 chars)**: Explain the rewardPerToken accumulator pattern used in Synthetix staking.

**Keywords**: rewardPerToken, accumulator, staking, proportional, O(1)

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 3/5 | 60% | **RewardPerToken (RP) accumulator pattern** is a simple yet effective way to track and update the reward per token (RP) for a Synthetix staking protocol. It's a simple accumulator that accumulates a f |
| DPO | 3/5 | 60% | **RewardPerToken (RP) accumulator** is a simple accumulator pattern used in Synthetix staking protocols to calculate the total reward per token (RPT) over a given period. The pattern is as follows:  1 |

### Example 15 — `protocol_design`

**Prompt (first 200 chars)**: What are the tradeoffs of deploying on Ethereum L1 vs an L2 rollup?

**Keywords**: L1, L2, rollup, sequencer, gas, finality

| Model | Hits | Score | Response (first 200 chars) |
|---|---|---|---|
| SFT | 4/6 | 67% | **Ethereum L1 (Mainnet)**: - **Decentralization**: L1 is a fully decentralized network, with no central authority or oracle. - **Low latency**: transactions are confirmed and broadcast instantly, redu |
| DPO | 3/6 | 50% | **Ethereum L1 (Mainnet)**:  1. **Decentralization**: L1 is a fully decentralized network, with no central authority or oracle. This means that rollup nodes are completely isolated from each other and  |
