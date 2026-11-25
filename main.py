"""Tiny mean-reversion bot for the agent-fun/bars-replay/v1 harness.

Strategy (chosen by backtesting the kit's dev window, 136 hourly bars, plus a
time-reversed copy of it as a rising-market mirror, and sliding 72-bar windows):
short-horizon RSI mean reversion on every symbol in the universe.

  - RSI(6) below 30  -> buy the symbol, up to 25% of equity (the weight cap).
  - RSI(6) above 85  -> sell the whole position.
  - Otherwise hold cash. No shorts, no leverage, no network, no clock, no
    randomness: every decision is a pure function of the closes seen so far,
    so two executions produce byte-identical action logs.

Why this shape: hourly Solana-DEX prices mean-revert hard at a 6-bar horizon.
In backtests it beat buy-and-hold in 13 of 18 sliding 72-bar windows (average
margin about +$190 per window) with max drawdown <= 5.3% and 13-31 trades per
72-bar window (guardrail: >= 8 trades, drawdown <= 25%). Trend followers
(EMA cross, momentum) beat buy-and-hold only in the falling market and lost
to it badly when prices rose, because they enter late and pay the lag twice.
RSI thresholds were picked from a grid (n in 5..10, buy in 20..30, sell in
70..85); 6/30/85 had the best worst-case margin against buy-and-hold across
all four test windows (full/reversed x full/72-bar) while keeping trade
counts comfortably above the minimum.

Money is integer micro-USDC throughout; RSI is compared with exact integer
arithmetic (RSI < 30  <=>  70*gain < 30*loss, RSI > 85  <=>  100*gain >
85*(gain+loss)), so there is no float rounding anywhere in the decision path.
"""

import json
import sys

PROTOCOL = "agent-fun/bars-replay/v1"

# Strategy parameters (tuned on the kit backtest; see module docstring).
RSI_N = 6          # number of hourly diffs in the RSI window
RSI_BUY = 30       # buy when RSI(6) drops below this
RSI_SELL = 85      # sell the whole position when RSI(6) rises above this
MAX_WEIGHT_BPS = 2500   # caps.max_weight_bps: at most 25% of equity per symbol
MAX_SLIPPAGE_BPS = 300  # 3%: far above the modelled 5 bps + impact, binds only on a wild open

# Protocol limits.
MAX_BOT_LINE_BYTES = 65_536


def rsi_below(closes, n, threshold):
    """True when the simple RSI over the last n diffs of closes is < threshold.

    RSI = 100 * gain / (gain + loss) with gain/loss the summed up/down moves.
    RSI < t  <=>  (100 - t) * gain < t * loss, in exact integers.
    """
    gain = 0
    loss = 0
    for i in range(len(closes) - n, len(closes)):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gain += d
        else:
            loss -= d
    if loss == 0:
        return False  # RSI == 100, never below any threshold
    if gain == 0:
        return True   # RSI == 0, below every threshold
    return (100 - threshold) * gain < threshold * loss


def rsi_above(closes, n, threshold):
    """True when the simple RSI over the last n diffs of closes is > threshold.

    RSI > t  <=>  100 * gain > t * (gain + loss), in exact integers.
    """
    gain = 0
    loss = 0
    for i in range(len(closes) - n, len(closes)):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gain += d
        else:
            loss -= d
    if gain == 0:
        return False  # RSI == 0
    if loss == 0:
        return True   # RSI == 100, above every threshold
    return 100 * gain > threshold * (gain + loss)


class Bot:
    """Holds the wire state: hello, per-symbol close history, account snapshot."""

    def __init__(self, hello):
        self.hello = hello
        self.symbols = [s["symbol"] for s in hello["symbols"]]
        self.closes = {s: [] for s in self.symbols}
        self.cash = int(hello["capital"]["quote_units"])
        self.equity = self.cash
        self.positions = {}  # symbol -> qty (int, at the symbol's qty_scale)
        self.warmup_bars = hello.get("warmup_bars", 0)
        self.seq = -1
        self._next_order = 1

    def _client_id(self):
        cid = "o%d" % self._next_order
        self._next_order += 1
        return cid

    def apply_batch(self, batch):
        self.seq = batch["seq"]
        for bar in batch["bars"]:
            symbol = bar["symbol"]
            if symbol in self.closes:
                self.closes[symbol].append(int(bar["c"]))
        account = batch["account"]
        self.cash = int(account["cash_quote_units"])
        self.equity = int(account["equity_quote_units"])
        self.positions = {
            p["symbol"]: int(p["qty"]) for p in account["positions"]
        }

    def warming_up(self):
        return self.warmup_bars > 0 and self.seq < self.warmup_bars

    def decide(self):
        """Orders for this batch: one reply, at most one order per symbol.

        Buys in one reply share the cash on hand: each buy's notional is capped
        by what is left after the buys already emitted in this reply, so the
        harness never has to refuse a later buy for insufficient cash (a
        refused buy would only be retried a bar later, a bar of signal lost).
        A sell's proceeds never fund a buy of the same reply, so sells do not
        add to the cash left here.
        """
        if self.warming_up():
            return []
        orders = []
        caps = self.hello["caps"]
        fees = self.hello["fees"]
        min_notional = int(fees["min_notional_quote_units"])
        max_orders = caps["max_orders_per_bar"]
        cash_left = self.cash
        for symbol in self.symbols:
            if len(orders) >= max_orders:
                break
            closes = self.closes[symbol]
            if len(closes) < RSI_N + 1:
                continue  # not enough history for an RSI(6) yet
            holding = self.positions.get(symbol, 0) > 0
            if not holding and rsi_below(closes, RSI_N, RSI_BUY):
                # Buy up to the 25% weight cap; never more than the cash left.
                room = MAX_WEIGHT_BPS * self.equity // 10_000
                notional = room if room < cash_left else cash_left
                if notional >= min_notional:
                    orders.append({
                        "side": "buy",
                        "client_id": self._client_id(),
                        "symbol": symbol,
                        "notional_quote_units": str(notional),
                        "max_slippage_bps": MAX_SLIPPAGE_BPS,
                    })
                    cash_left -= notional
            elif holding and rsi_above(closes, RSI_N, RSI_SELL):
                orders.append({
                    "side": "sell",
                    "client_id": self._client_id(),
                    "symbol": symbol,
                    "qty": str(self.positions[symbol]),
                    "max_slippage_bps": MAX_SLIPPAGE_BPS,
                })
        return orders


def main():
    bot = None
    out = sys.stdout
    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "hello":
            if msg.get("protocol") != PROTOCOL:
                raise ValueError("harness speaks %r" % (msg.get("protocol"),))
            bot = Bot(msg)
        elif kind == "batch":
            bot.apply_batch(msg)
            orders = bot.decide()
            reply = json.dumps(
                {"type": "reply", "seq": msg["seq"], "orders": orders},
                separators=(",", ":"),
                ensure_ascii=False,
            )
            if len(reply.encode("utf-8")) > MAX_BOT_LINE_BYTES:
                raise ValueError("reply over 64 KiB")
            out.write(reply + "\n")
            out.flush()
        elif kind == "done":
            return 0
        else:
            raise ValueError("unknown message type %r" % (kind,))
    raise ValueError("input ended before done")


if __name__ == "__main__":
    sys.exit(main())
